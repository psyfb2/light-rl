import time
import gym
import torch
import torch.nn
import torch.multiprocessing as mp
import numpy as np 
import scipy.stats

from torch.distributions.normal import Normal
from tqdm import tqdm

from light_rl.common.nets.fc_network import FCNetwork
from light_rl.common.nets.device import DEVICE
from light_rl.common.base.feature_transform import Transform
from light_rl.common.base.agent import BaseAgent


class ES(BaseAgent):
    def __init__(self, action_space: gym.Space, state_space: gym.Space,
            ft_transformer=Transform(), actor_hidden_layers=[],
            lstm_hidden_dim=256, num_workers=mp.cpu_count(),
            lr=0.1, pop_size=50, std_noise=0.5, vbn_states: list = None,
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
            vbn_states (List[np.array], optional): states to use for virtual batch normalization. These
                states should be gathered by playing the enviroment with random actions and keeping
                each encoutered state with a probability of 1%. Then once 128 states have been gathered
                this can be the vbn_states. If None will not use virtual batch normalization.
                Defaults to None.
            reward_shaping_method (str, optional): method to use to shape pop_size rewards.
                can be 'rank' for rank transformation, 'standardise' to standardise the list
                of rewards. Defaults to 'rank'
        """
        super().__init__(action_space, state_space)
        self.init_kwargs = {k: v for k, v in locals().items() if k != "self"}
        
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

        self.action_size = action_space.shape[0]
        self.state_size = ft_transformer.transform(np.zeros(state_space.shape)).shape[0]

        self.actor_net = FCNetwork(
            (self.state_size, *actor_hidden_layers, self.action_size ), 
            lstm_hidden_size=lstm_hidden_dim,
            output_activation=torch.nn.Tanh 
        )
        self.optim = torch.optim.Adam(self.actor_net.parameters(), lr=lr)

        # put the actor in shared memory
        # self.actor_net.share_memory()
    
    def get_action(self, s: np.ndarray, explore: bool = False, rec_state=None):
        with torch.no_grad():
            s = self._transform_state(s)
            a, rec_state = self.actor_net((s, rec_state))
        return a.numpy(), rec_state
    
    def _transform_state(self, s: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            self.ft_transformer.transform(s)
        ).to(DEVICE).float()
    
    def _generate_noise(self, net: torch.nn.Module) -> list:
        """ Get guassian noise N(0, 1) for each parameter in the net.

        Args:
            net (torch.nn.Module): net to generate noise for

        Returns:
            List[np.ndarray]: list of noise matricies which can be used to
                perturb the net.
        """
        param_noise = []
        for p in net.parameters():
            noise = np.random.normal(size=tuple(p.data.shape))
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
        if self.reward_shaping_method == 'rank:':
            ranked = scipy.stats.rankdata(rewards)
            norm = (ranked - 1) / (len(ranked) - 1)
            norm -= 0.5
            return norm
        else:
            return (rewards - np.mean(rewards)) / np.std(rewards)

    def train(self, env: gym.Env, max_timesteps: int, max_training_time: float, 
              target_return: float, max_episode_length: int, eval_freq: int, eval_episodes: int):
        episode_rewards = []
        episode_r_times = []
        start_time = time.time()
        timesteps_elapsed = 0
        timestep_last_eval = 0

        with tqdm(total=max_timesteps) as progress_bar:
            while timesteps_elapsed < max_timesteps:
                # play pop_size episodes
                fs = []
                perturbs = []

                for i in range(self.pop_size // 2):
                    old_net = self.actor_net.state_dict()
                    perturb = self._generate_noise(self.actor_net)

                    for pert in (perturb, [-arr for arr in perturb]): # mirror sampling
                        self._perturb_net(self.actor_net, pert)
                        reward, iters = self.play_episode(env, max_episode_length)

                        fs.append(reward)
                        perturbs.append(pert)
                        self.actor_net.load_state_dict(old_net)
                        progress_bar.update(iters)
                        timesteps_elapsed += iters

                        if len(episode_rewards) < 1e5:
                            episode_rewards.append(reward)
                            episode_r_times.append(time.time() - start_time)
                
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
        
        return episode_rewards, episode_r_times



'''
# hyperparams
lr = 0.1
pop_size = 50
std_noise = 0.5

assert pop_size % 2 == 0


# func params
a = 1
b = 1
c = 2
d = -1
const = 1

assert a > 0
assert b > 0  # must be positive so that maxima exists

maxima = (c / (2 * a), d / (2 * b))  # from calculus

def fitness(theta: np.ndarray):
    # multidimensional quadratic
    x, y = theta
    return -a*x**2 - b*y**2 + c*x + d*y + const


def evolution_strategies(theta: np.ndarray, epochs=10000):
    assert theta.ndim == 1


    for epoch in range(epochs):
        # sample e1, ..., en, where ei ~ N(0, I)
        noises = np.random.randn(pop_size // 2, 2)
        noises = np.concatenate((noises, -noises))

        # compute returns
        returns = np.array([fitness(theta + std_noise * noise) for noise in noises])
        returns = (returns - np.mean(returns)) / np.std(returns)

        # update theta
        theta = theta + lr * (np.dot(noises.T, returns) / (pop_size * std_noise))
    
    return theta


print("True Maxima:", maxima)
print("Estimated Maxima:", evolution_strategies(np.array([10, 10])))

'''
