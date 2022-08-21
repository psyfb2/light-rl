import gym
import numpy as np
import matplotlib.pyplot as plt
import torch.nn
import os

from torch.optim import Adam
from torch.distributions.normal import Normal
from gym import wrappers
from tqdm import tqdm
from copy import deepcopy

from agent import Agent
from plotting import plot_avg_reward
from fc_network import FCNetwork, DEVICE
from continious_pg_mountaincar import Transform, RBF_Transform
from experience_replay import Transition, ReplayBuffer


PENDULUM_CONFIG = {
    "env": "Pendulum-v1",
    "target_return": -300.0,
    "episode_length": 200,
    "gamma": 0.99,
    "save_filename": "pendulum_latest.pt",

    "policy_learning_rate": 1e-3,
    "critic_learning_rate": 1e-3,
    "critic_hidden_size": [64, 64],
    "policy_hidden_size": [64, 64],
    "tau": 0.01,
    "batch_size": 64,
    "buffer_capacity": int(1e6),
}


class DDPG(Agent):
    def __init__(self, env: gym.Env, ft_transformer: Transform, 
            actor_hidden_layers=[], critic_hidden_layers=[], 
            actor_lr=1e-4, critic_lr=1e-4, noise_std=0.1, gamma=1.0, tau=0.01):
        """
        DDPG, assumes continious actions.
        """
        self.env = env
        self.ft_transformer = ft_transformer
        self.gamma = gamma
        self.tau = tau

        # init models
        self.state_dim = self._transform_state(env.reset()).shape[-1]
        self.action_dim = env.action_space.shape[-1]
        self.upper_action_bound = env.action_space.high[0]
        self.lower_action_bound = env.action_space.low[0]

        self.actor_net = FCNetwork(
            (self.state_dim, *actor_hidden_layers, self.action_dim), torch.nn.Tanh 
        )
        self.actor_target_net = FCNetwork(
            (self.state_dim, *actor_hidden_layers, self.action_dim), torch.nn.Tanh 
        )
        self.actor_target_net.hard_update(self.actor_net)

        self.actor_optim = Adam(
            self.actor_net.parameters(), lr=actor_lr, eps=1e-3
        )

        self.critic_net = FCNetwork(
            (self.state_dim + self.action_dim, *critic_hidden_layers, 1)
        )
        self.critic_target_net = FCNetwork(
            (self.state_dim + self.action_dim, *critic_hidden_layers, 1)
        )
        self.critic_target_net.hard_update(self.critic_net)
        self.critic_optim = Adam(
            self.critic_net.parameters(), lr=critic_lr, eps=1e-3
        )

        self.noise = Normal(torch.tensor([0.0]), torch.tensor([noise_std]))
    
    def _transform_state(self, s: np.ndarray) -> torch.Tensor:
        if len(s.shape) == 1:
            transformed = self.ft_transformer.transform([s])[0]
        else:
            transformed = self.ft_transformer.transform(s)
        return torch.from_numpy(transformed).to(DEVICE).float()
    
    def get_action(self, s: np.ndarray, explore: bool = False):
        with torch.no_grad():
            s = self._transform_state(s)
            a = self.actor_net(s)
            if explore:
                a += torch.reshape(self.noise.sample(a.shape), a.shape)
        return a.numpy()

    def learn(self, batch: Transition, max_grad_norm=50):
        '''
        batch is tuple of (s, a, r, next_s, terminal)
        where 
            s.shape =>          (batch_size, state_dim)
            a.shape =>          (batch_size, action_dim)
            r.shape =>          (batch_size, 1)
            next_s.shape =>     (batch_size, state_dim)
            terminal.shape =>   (batch_size, 1)
        '''
        q_loss = torch.tensor([0.0])
        p_loss = torch.tensor([0.0])
        batch_size = batch.states.shape[0]

        batch = Transition(
            self._transform_state(batch.states), 
            torch.from_numpy(batch.actions).to(DEVICE), 
            self._transform_state(batch.next_states), 
            torch.from_numpy(batch.rewards).to(DEVICE), 
            torch.from_numpy(batch.terminal).to(DEVICE)
        )

        # first update critic by minimising MSE of TD error on batch
        with torch.no_grad():
            q_values =  self.critic_target_net(
                torch.concat(
                    (batch.next_states, self.actor_target_net(batch.next_states)), 
                    axis=-1
                )
            )
            targets = batch.rewards + (self.gamma * (1 - batch.terminal) * q_values) 
        
        y_pred = self.critic_net(torch.concat((batch.states, batch.actions), axis=-1))
        q_loss = (targets - y_pred).pow(2).sum() / batch_size

        self.critic_optim.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_grad_norm)
        self.critic_optim.step()

        # now update actor by maxmizing q value from critic
        q_values = self.critic_net(
            torch.concat((batch.states, self.actor_net(batch.states)), axis=-1)
        )
        p_loss = (-q_values).sum() / batch_size

        self.actor_optim.zero_grad()
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_grad_norm)
        self.actor_optim.step()

        # update the target networks
        self.critic_target_net.soft_update(self.critic_net, self.tau)
        self.actor_target_net.soft_update(self.actor_net, self.tau)
    
    def train_agent(self, n_episodes=150, max_grad_norm=50, buffer_capacity=int(1e6), batch_size=64, max_episode_length=200, eval_freq=10, eval_episodes=3, target_return=-300) -> list:
        replay_buffer = ReplayBuffer(buffer_capacity)
        episode_rewards = np.zeros((n_episodes, ))

        for i in tqdm(range(n_episodes)):
            done = False
            self.reset()
            s = self.env.reset()
            episode_reward = 0
            iters = 0

            while not done and iters < max_episode_length:
                a = self.get_action(s, True)
                next_s, r, terminal, truncated, info = self.env.step(a)
                done = terminal

                replay_buffer.push(
                    np.array(s, dtype=np.float32),
                    np.array(a, dtype=np.float32),
                    np.array(next_s, dtype=np.float32),
                    np.array([r], dtype=np.float32),
                    np.array([terminal], dtype=np.float32),
                )
                if len(replay_buffer) >= batch_size:
                    batch = replay_buffer.sample(batch_size)
                    self.learn(batch, max_grad_norm)

                s = next_s
                episode_reward += r
                iters += 1
            
            episode_rewards[i] = episode_reward
            
            if (i + 1) % eval_freq == 0:
                avg_r = 0
                for _ in range(eval_episodes):
                    avg_r += play_episode(self.env, self, max_episode_length)
                avg_r /= eval_episodes

                tqdm.write(f"Episode {i + 1}: Average reward using {eval_episodes} evals: {avg_r}")

                if avg_r >= target_return:
                    tqdm.write(f"Avg reward {avg_r} >= Targer reward {target_return}, stopping early")
                    break
        
        return episode_rewards


def play_episode(env: gym.Env, agent: Agent, max_episode_length=200):
    done = False
    agent.reset()
    s = env.reset()
    episode_reward = 0
    iters = 0

    while not done and iters < max_episode_length:
        a = agent.get_action(s)
        next_s, r, terminal, truncated, info = env.step(a)
        done = terminal

        s = next_s
        episode_reward += r
        iters += 1
    
    return episode_reward

def main(n_episodes=500):
    config = PENDULUM_CONFIG
    env =  gym.make(config["env"], new_step_api=True)

    #import pickle
    #with open('rbf_transformer.pkl', 'rb') as handle:
        #rbf = pickle.load(handle)

    agent = DDPG(env, Transform(), config["policy_hidden_size"], config["critic_hidden_size"], config["policy_learning_rate"], config["critic_learning_rate"], 0.1, config["gamma"], config["gamma"])
    rewards = agent.train_agent(n_episodes, 50, config["buffer_capacity"], config["batch_size"])
    plt.plot(rewards)
    plt.show()
    plot_avg_reward(rewards)

    env = wrappers.RecordVideo(env, 'ddpg_car_vids', new_step_api=True)
    play_episode(env, agent)


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    main()