import time
import gym
import torch
import torch.nn
import torch.multiprocessing as mp
import numpy as np 
import scipy.stats
import torch.multiprocessing as mp


from torch.distributions.normal import Normal
from tqdm import tqdm

from light_rl.common.nets.fc_network import FCNetwork
from light_rl.common.nets.fc_network_vbn import FCNetworkVBN
from light_rl.common.base.feature_transform import Transform
from light_rl.common.base.agent import BaseAgent

DEVICE = "cpu"  # ES only supports CPU training

class ES(BaseAgent):
    def __init__(self, action_space: gym.Space, state_space: gym.Space,
            ft_transformer=Transform(), actor_hidden_layers=[],
            lstm_hidden_dim=256, num_workers=mp.cpu_count(),
            lr=0.1, pop_size=50, std_noise=0.5, vbn_states: torch.Tensor = None,
            reward_shaping_method='rank'):
        """ Constructor for Evolution Strategies (ES).
        Action space only continious for now (TODO: allow discrete action spaces)

        Args:
            action_space (gym.Space): action space for this agent
            state_space (gym.Space): state space for this agent
            ft_transformer (Transform): feature transformer.
            actor_hidden_layers (list, optional): actor hidden layers. 
                can specify int for Linear followed by ReLU or
                'lstm' to use an lstm layer. Defaults to [].
            lstm_hidden_dim (int, optional): size of hidden and context vector in lstm
                if one is used. Defaults to 256.
            num_workers (int, optional): number of workers to run the fitness function of ES in parallel.
                Defaults to mp.cpu_count().
            lr (float, optional): learning rate. Defaults to 0.1.
            pop_size (int, optional): population size, if odd will +1 to make even. Defaults to 50.
            std_noise (float, optional): noise standard deviation for creating offspring. Defaults to 0.5.
            vbn_states (torch.Tensor, optional): states to use for virtual batch normalization. These
                can be gathered by using the sample_vbn_states() method of this class. If None will
                not use VBN. Defaults to None.
            reward_shaping_method (str, optional): method to use to shape pop_size rewards.
                can be 'rank' for rank transformation, 'standardise' to standardise the list
                of rewards. Defaults to 'rank'
        """
        super().__init__(action_space, state_space)
        self.init_kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        
        self.ft_transformer = ft_transformer
        self.actor_hidden_layers = actor_hidden_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_workers = num_workers
        self.lr = lr
        # pop size should be even for mirrow sampling
        self.pop_size = pop_size if pop_size % 2 == 0 else pop_size + 1
        self.std_noise = std_noise
        self.vbn_states = vbn_states
        self.reward_shaping_method = reward_shaping_method
        self.MAX_SEED = 2**32 - 1

        self.action_size = action_space.shape[0]
        self.state_size = ft_transformer.transform(np.zeros(state_space.shape)).shape[0]

        if self.vbn_states is None:
            self.actor_net = FCNetwork(
                (self.state_size, *actor_hidden_layers, self.action_size ), 
                lstm_hidden_size=lstm_hidden_dim,
                output_activation=torch.nn.Tanh 
            )
        else:
            self.actor_net = FCNetworkVBN(vbn_states,
                (self.state_size, *actor_hidden_layers, self.action_size ), 
                lstm_hidden_size=lstm_hidden_dim,
                output_activation=torch.nn.Tanh 
            )
            
        self.optim = torch.optim.Adam(self.actor_net.parameters(), lr=lr)
        # put the global actor in shared memory
        self.actor_net.share_memory()

        self.torch_saveables.update(
            {
                "actor_net": self.actor_net,
                "optim": self.optim,
            }
        )
    
    
    def get_action(self, s: np.ndarray, explore: bool = False, rec_state=None):
        with torch.no_grad():
            s = self._transform_state(s)
            a, rec_state = self.actor_net((s, rec_state))
        return a.numpy(), rec_state
    
    def _transform_state(self, s: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            self.ft_transformer.transform(s)
        ).to(DEVICE).float()
    
    def _generate_noise(self, net: torch.nn.Module, rng: np.random.RandomState) -> list:
        """ Get guassian noise N(0, 1) for each parameter in the net.

        Args:
            net (torch.nn.Module): net to generate noise for
            rng (np.random.RandomState): random number generator used to generator noise

        Returns:
            List[np.ndarray]: list of noise matricies which can be used to
                perturb the net.
        """
        param_noise = []
        for p in net.parameters():
            noise = rng.normal(size=tuple(p.data.shape))
            param_noise.append(noise)
        return param_noise
    
    def _perturb_net(self, net: torch.nn.Module, param_noise: list):
        """ Perturb a net inplace by adding guassian noise N(0, std_noise)
        to each parameter.

        Args:
            net (torch.nn.Module): net to perturb inplace.
            param_noise (list): guassian Noise(0, 1) for each parameter.
                comes from self._generate_noise(net).
        """
        for noise, p in zip(param_noise, net.parameters()):
            p.data += torch.from_numpy(self.std_noise * noise).to(DEVICE).float()
    
    def _shape_rewards(self, rewards: np.ndarray):
        """ Perform reward shaping to a list of rewards

        Args:
            rewards (np.ndarray): rewards to shape

        Returns:
            np.ndarray: shaped rewards
        """
        if self.reward_shaping_method == 'rank':
            ranked = scipy.stats.rankdata(rewards)
            norm = (ranked - 1) / (len(ranked) - 1)
            norm -= 0.5
            return norm
        else:
            return (rewards - np.mean(rewards)) / np.std(rewards)
        
    def _es_worker(self, rank: int, env: gym.Env, max_episode_length: int,
            results_q: mp.Queue, task_q: mp.Queue):
        """ ES worker. Waits for True in task_q. Will then sync local net with the
        global actor net and add positive and negative (mirror sampling)
        guassian noise std_noise*N(0, 1) to the local net. Then will play an episode
        for the positive and negative sample. Then put results in the results_q. 
        if False is received on the task_q will stop listenining on the queue and terminate.

        Args:
            rank (int): id of this process
            env (gym.Env): enviroment used for fitness evaluation (fitness is reward from one episode)
            max_episode_length (int): maximum episode length when evaluating fitness
            results_q (mp.Queue): Queue to put (seed, pos_reward, neg_reward, timesteps) results
            task_q (mp.Queue): Listens on this queue for (True, ) in which case will 
                generate a random seed, and seed positive then negative samples.
                Then after playing episodes will put 
                (seed, pos_reward, pos_length, pos_time, neg_reward, neg_length, neg_time)
                on this queue.
        """
        if self.vbn_states is None:
            local_actor_net = FCNetwork(
                (self.state_size, *self.actor_hidden_layers, self.action_size ), 
                lstm_hidden_size=self.lstm_hidden_dim,
                output_activation=torch.nn.Tanh 
            )
        else:
            local_actor_net = FCNetworkVBN(self.vbn_states,
                (self.state_size, *self.actor_hidden_layers, self.action_size ), 
                lstm_hidden_size=self.lstm_hidden_dim,
                output_activation=torch.nn.Tanh 
            )
            
        while task_q.get():
            # generate random noise with seed
            seed = np.random.randint(0, self.MAX_SEED + 1, dtype=np.uint32)
            rng = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
            perturb = self._generate_noise(self.actor_net, rng)

            # evaluate fitness of new parameters
            rewards = []
            lengths = []
            times = []
            for pert in (perturb, [-arr for arr in perturb]): # mirror sampling
                # sync local net with global net
                local_actor_net.load_state_dict(self.actor_net.state_dict())
                self._perturb_net(local_actor_net, pert)

                # play episode with local actor net
                temp = self.actor_net
                self.actor_net = local_actor_net
                reward, iters = self.play_episode(env, max_episode_length)
                self.actor_net = temp

                rewards.append(reward)
                lengths.append(iters)
                times.append(time.time())
                
            results_q.put((seed, 
                rewards[0], lengths[0], times[0], 
                rewards[1], lengths[1], times[1])
            )

    def train(self, env: gym.Env, max_timesteps: int, max_training_time: float, 
              target_return: float, max_episode_length: int, eval_freq: int, eval_episodes: int):
        episode_rewards = []
        episode_r_times = []
        start_time = time.time()
        timesteps_elapsed = 0
        timestep_last_eval = 0

        # init workers
        results_q = mp.SimpleQueue()
        task_q = mp.SimpleQueue()
        processes = []

        for rank in range(self.num_workers):
            p = mp.Process(target=self._es_worker, args=(
                    rank, env, max_episode_length, results_q, task_q
                )
            )
            p.start()
            processes.append(p)

        with tqdm(total=max_timesteps) as progress_bar:
            while timesteps_elapsed < max_timesteps:
                # play pop_size episodes
                fs = []
                perturbs = []

                for _ in range(self.pop_size // 2):
                    task_q.put(True)
                
                for _ in range(self.pop_size // 2):
                    seed, pos_r, pos_iters, pos_t, neg_r, neg_iters, neg_t = results_q.get()
                    result = ((pos_r, pos_iters, pos_t), (neg_r, neg_iters, neg_t))

                    # reconstruct noise and add to perturbs
                    rng = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
                    perturb = self._generate_noise(self.actor_net, rng)

                    # update fs, perturbs
                    for i, pert in enumerate((perturb, [-arr for arr in perturb])): # mirror sampling
                        reward, iters, t = result[i]

                        fs.append(reward)
                        perturbs.append(pert)
                        progress_bar.update(iters)
                        timesteps_elapsed += iters

                        if len(episode_rewards) < 1e5:
                            episode_rewards.append(reward)
                            episode_r_times.append(t - start_time)

                fs = self._shape_rewards(fs)
                # update actor_net according to ES update
                self.optim.zero_grad()
                for idx, p in enumerate(self.actor_net.parameters()):
                    weight_update = np.zeros(p.data.shape)

                    for n in range(len(fs)):
                        weight_update += fs[n] * perturbs[n][idx]

                    weight_update = weight_update / (self.pop_size * self.std_noise)
                    p.grad = torch.from_numpy(-weight_update).to(DEVICE).float()
                self.optim.step()

                elapsed_time = time.time() - start_time
                if timesteps_elapsed - timestep_last_eval > eval_freq:
                    timestep_last_eval = timesteps_elapsed
                    avg_r = 0
                    for _ in range(eval_episodes):
                        avg_r += self.play_episode(env, max_episode_length)[0]
                    avg_r /= eval_episodes

                    progress_bar.write(
                        f"Step {timesteps_elapsed} at time {round(elapsed_time, 3)}s. "
                        f"Average reward using {eval_episodes} evals: {avg_r}"
                    )

                    if avg_r >= target_return:
                        progress_bar.write(f"Avg reward {avg_r} >= Target reward {target_return}, stopping early.")
                        break
                
                if elapsed_time > max_training_time:
                    progress_bar.write(f"Training time limit reached. Training stopped at {elapsed_time}s.")
                    break

        # termiante the workers
        for _ in range(self.num_workers):
            task_q.put(False)
        for p in processes:
            p.join()

        return episode_rewards, episode_r_times
    
    @staticmethod
    def sample_vbn_states(env: gym.Env, num_samples=128, keep_prob=0.01) -> torch.Tensor:
        """ sample num_samples states from env by performing random action.
        each state is kept with keep_prob until num_samples states have been saved.
        The returns num_samples states as a torch tensor with shape 
        (num_samples, observation_size)

        Args:
            env (gym.Env): Enviroment that follows two_step gym api
            num_samples (int, optional): number of states to collect. Defaults to 128.
            keep_prob (float, optional): probability of keeping an observed state. 
                Defaults to 0.01.
        """
        states = []
        while len(states) < num_samples:
            done = False
            s = env.reset()

            while not done:
                a = env.action_space.sample()
                s, r, terminal, truncated, _ = env.step(a)
                done = terminal or truncated

                if np.random.rand() < keep_prob:
                    states.append(s)

                    if len(states) >= num_samples:
                        break
        
        return torch.tensor(np.array(states))
