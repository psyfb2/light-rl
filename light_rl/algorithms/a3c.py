import time
import gym
import torch
import numpy as np

from torch.optim import Adam
from torch.distributions.normal import Normal

from light_rl.common.base.agent import BaseAgent
from light_rl.common.nets.device import DEVICE
from light_rl.common.nets.fc_network import FCNetwork, LSTM_STR
from light_rl.common.base.feature_transform import Transform
from light_rl.common.optimizers.shared_adam import SharedAdam
from light_rl.algorithms.vanilla_policy_gradient import VanillaPolicyGradient


import gym
import numpy as np
import matplotlib.pyplot as plt
import torch.nn
import torch.multiprocessing as mp


from gym import wrappers
from tqdm import tqdm
from copy import deepcopy



class A3C(VanillaPolicyGradient):
    def __init__(self, action_space: gym.Space, state_space: gym.Space,
            ft_transformer: Transform, 
            actor_hidden_layers=[], critic_hidden_layers=[], 
            actor_lr=1e-4, critic_lr=1e-4, actor_adam_eps=1e-3, 
            critic_adam_eps=1e-3, gamma=0.999, max_grad_norm=50,
            lstm_hidden_dim=256, tmax=20,
            entropy_beta=1e-5, num_workers=mp.cpu_count(), 
            shared_optimizer=True):
        """ Constructor for A3C.
        Action space only continious for now (TODO: allow discrete action spaces)
        Uses spherical covariance for normal distribution with continious actions.

        Args:
            action_space (gym.Space): action space for this agent
            state_space (gym.Space): state space for this agent
            ft_transformer (Transform): feature transformer.
            actor_hidden_layers (list, optional): actor hidden layers. Defaults to [].
            critic_hidden_layers (list, optional): critic hidden layers. Defaults to [].
            actor_lr (_type_, optional): actor learning rate. Defaults to 1e-4.
            critic_lr (_type_, optional): critic learning rate. Defaults to 1e-4.
            gamma (float, optional): discount factor gamma. Defaults to 0.999.
            max_grad_norm (int, optional): max grad norm used in gradient clipping. Defaults to 50.
            lstm_hidden_dim (int, optional): (hx, cx) vector sizes of lstm if one is used
            tmax (int, optional): n used in n-step-return (if 1 would be TD(0)). Defaults to 20.
            entropy_beta (float, optional): constant multiplier on exploration entropy term. Defaults to 1e-5.
            num_workers (int, optional): number of workers used to train A3C in parallel.
            shared_optimizer (bool, optional): Whether workers should use shared optimizer. Defaults to True.
        """
        super().__init__(action_space, state_space,
            ft_transformer, actor_hidden_layers, critic_hidden_layers, 
            actor_lr, critic_lr, actor_adam_eps, critic_adam_eps, 
            gamma, max_grad_norm, lstm_hidden_dim)
        
        self.actor_hidden_layers = actor_hidden_layers
        self.critic_hidden_layers = critic_hidden_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.tmax = tmax
        self.entropy_beta = entropy_beta
        self.num_workers = num_workers

        if shared_optimizer:
            self.actor_optim = SharedAdam(
                self.actor_net.parameters(), lr=actor_lr
            )
            self.actor_optim.share_memory()

            self.critic_optim = SharedAdam(
                self.critic_net.parameters(), lr=critic_lr
            )
            self.critic_optim.share_memory()

        # put the actor and critic params in shared memory
        # (grads are None and so won't be put in shared memory)        
        self.actor_net.share_memory()
        self.critic_net.share_memory()
    
    def _ensure_shared_grads(self, model, shared_model):
        for param, shared_param in zip(model.parameters(),
                                       shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
    
    def _a3c_worker(self, rank: int, env: gym.Env, max_timesteps: int,
                    max_training_time: float, target_return: float, 
                    max_episode_length: int, eval_freq: int, eval_episodes: int,
                    timesteps_counter: mp.Value, lock: mp.Lock, return_q: mp.Queue):
        env = deepcopy(self.env)

        local_actor_net = FCNetwork(
            # +1 in output for the standard deviation for At ~ N(means, std * Identity)
            (self.state_size, *self.actor_hidden_layers, self.action_size + 1), 
            lstm_hidden_size=self.lstm_hidden_dim
        )
        local_critic_net = FCNetwork(
            (self.state_size, *self.critic_hidden_layers, 1),
            lstm_hidden_size=self.lstm_hidden_dim
        )

        s = self._transform_state(env.reset())
        episode_reward = episode_length = 0
        episode_rewards = []
        episode_r_times = []
        start_time = time.time()
        actor_rec_state = critic_rec_state = None

        while True:
            with lock:
                if timesteps_counter.value >= max_timesteps:
                    break

            # sync local nets with global nets
            local_actor_net.load_state_dict(self.actor_net.state_dict())
            local_critic_net.load_state_dict(self.critic_net.state_dict())

            states = []    # s_t            for t in {t_start, ..., t_start + t_max - 1}
            log_probs = [] # log_prob(a_t)  for t in {t_start, ..., t_start + t_max - 1}
            entropies = [] # entropy(a_t)   for t in {t_start, ..., t_start + t_max - 1}
            rewards = []   # r_t            for t in {t_start + 1, ..., t_start + t_max}

            for step in range(self.tmax):
                # get action
                actor_out, actor_rec_state = local_actor_net((s, actor_rec_state)) # shape => (self.action_size + 1, )
                mu = actor_out[:self.action_size] # shape => (self.action_size, )
                std = self.soft_plus(actor_out[self.action_size:]) # shape => (1, )
                pdf = Normal(mu, std)
                a = pdf.sample()  # shape => (self.action_dim, )

                next_s, r, terminal, truncated, _ = env.step(a.detach().numpy())
                next_s = self._transform_state(next_s)

                log_probs.append(pdf.log_prob(a).sum())
                entropies.append(self.entropy_beta * torch.log(std))
                states.append(s)
                rewards.append(r)

                episode_reward += r
                episode_length += 1
                s = next_s
                elapsed_time = time.time() - start_time

                with lock:
                    timesteps_counter.value += 1

                if terminal or episode_length >= max_episode_length:
                    episode_rewards.append(episode_reward)
                    episode_r_times.append(elapsed_time)
                    s = self._transform_state(env.reset())
                    episode_reward = episode_length = 0
                    print(episode_rewards[-1])
                    break
            
            if terminal: 
                R = 0
            else:
                # bootstrap for n-step reward
                with torch.no_grad():
                    R = local_critic_net((next_s))

            actor_loss = critic_loss = 0

            for i in range(len(states) - 1, -1, -1):
                R = rewards[i] + self.gamma * R
                td_error = R - local_critic_net(states[i])

                critic_loss += td_error.pow(2).sum()
                actor_loss -= td_error.detach() * log_probs[i] + entropies[i]
        
            # update global nets with accumulated gradients
            self.critic_optim.zero_grad()
            self.actor_optim.zero_grad()

            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_critic_net.parameters(), max_grad_norm)

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_actor_net.parameters(), max_grad_norm)

            self._ensure_shared_grads(local_actor_net, self.actor_net)
            self._ensure_shared_grads(local_critic_net, self.critic_net)

            self.critic_optim.step()
            self.actor_optim.step()
        
        return_q.put((rank, episode_rewards))

    def train(self, env: gym.Env, max_timesteps: int, max_training_time: float, 
              target_return: float, max_episode_length: int, eval_freq: int, eval_episodes: int):
        timesteps_counter = mp.Value('i', 0)
        return_q = mp.Queue()
        lock = mp.Lock()
        processes = []

        for rank in range(self.num_workers):
            p = mp.Process(target=self._a3c_worker, args=(
                    rank, env, max_timesteps, max_training_time, target_return, 
                    max_episode_length, eval_freq, eval_episodes,
                    timesteps_counter, lock, return_q
                )
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        episode_rewards = []
        for _ in range(self.num_workers):
            episode_rewards.append(return_q.get())
        episode_rewards.sort(key=lambda tpl : tpl[0])
        episode_rewards = [tpl[1] for tpl in episode_rewards]

        # list of list, each inner list episode reward overtime for worker
        return episode_rewards  