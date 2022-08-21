import gym
import numpy as np
import matplotlib.pyplot as plt
import torch.nn

from torch.optim import Adam
from torch.distributions.categorical import Categorical
from gym import wrappers
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
from copy import deepcopy

from agent import Agent
from plotting import plot_avg_reward
from fc_network import FCNetwork, DEVICE
from experience_replay import Transition, ReplayBuffer


CARTPOLE_CONFIG = {
    "eval_freq": 2000,
    "eval_episodes": 20,
    "learning_rate": 1e-4,
    "hidden_size": (128, ),
    "target_update_freq": 2000,
    "batch_size": 64,
    "buffer_capacity": int(1e6),
    "gamma": 0.99
}


class RBF_Transform:
    def __init__(self, env: gym.Env, n_episodes=5000, n_components=1000, gammas=(0.05, 0.1, 0.5, 1)):
        states = []
        for _ in range(n_episodes):
            s = env.reset()
            states.append(s)
            done = False
            while not done:
                a = env.action_space.sample()
                next_s, r, terminal, truncated, info = env.step(a)
                done = terminal or truncated

                states.append(next_s)
                s = next_s

        self.standard_scaler = StandardScaler()
        self.rbfs = FeatureUnion(
            [(f"rbf{g}", RBFSampler(gamma=g, n_components=n_components)) for g in gammas]
        )
        self.rbfs.fit(self.standard_scaler.fit_transform(states))
    
    def transform(self, s: np.ndarray) -> np.ndarray:
        # s.shape => (num_states, state_dim)
        return self.rbfs.transform(self.standard_scaler.transform(s))


class DQN(Agent):
    def __init__(self, env: gym.Env, ft_transformer: RBF_Transform,
            critic_hidden_layers=[], critic_lr=1e-4, gamma=1.0,
            batch_size=64, target_update_freq=2000, epsilon=0.05):
        """
        DQN Agent. Assumes 1D state and dicrete action space (0 to num_actions - 1)
        """
        self.env = env
        self.ft_transformer = ft_transformer
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.epsilon = epsilon

        # init dqn model
        self.state_dim = self._transform_state(env.reset()[None, :]).shape[-1]
        self.action_dim = env.action_space.n

        self.critic_net = FCNetwork(  # Q(s, a)
            (self.state_dim, *critic_hidden_layers, self.action_dim)
        )
        self.critic_target = deepcopy(self.critic_net)  # Q(s, a) used in td error calculation
        self.critic_optim = Adam(
            self.critic_net.parameters(), lr=critic_lr, eps=1e-3
        )
    
    def _transform_state(self, s: np.ndarray) -> torch.Tensor:
        # s.shape => (num_states, state_dim)
        return torch.from_numpy(
            self.ft_transformer.transform(s)
        ).to(DEVICE).float()
    
    
    def get_action(self, s: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        with torch.no_grad():
            s = self._transform_state(s[None, :])[0]
            a = torch.argmax(self.critic_net(s))
        return a.item()
    
    def learn(self, batch: Transition):
        '''
        batch is tuple of (s, a, r, next_s, terminal)
        where 
            s.shape =>          (batch_size, state_dim)
            a.shape =>          (batch_size, 1)
            r.shape =>          (batch_size, 1)
            next_s.shape =>     (batch_size, state_dim)
            terminal.shape =>   (batch_size, 1)
        '''
        with torch.no_grad():
            states = self._transform_state(batch.states)
            next_states = self._transform_state(batch.next_states)
            batch = Transition(
                states, 
                torch.from_numpy(batch.actions).to(DEVICE).to(torch.long), 
                next_states, 
                torch.from_numpy(batch.rewards).to(DEVICE), 
                torch.from_numpy(batch.terminal).to(DEVICE)
            )

            # calculate r + gamma * max Q(s', a')
            g = batch.rewards[:, 0] + (self.gamma * 
                (1 - batch.terminal[:, 0]) *
                torch.max(self.critic_target(batch.next_states), dim=1).values 
            )
            
        # update critic using MSE(g, Q(s, a))
        y_pred = self.critic_net(batch.states)[range(self.batch_size), batch.actions[:, 0]]
        loss = (g - y_pred).pow(2).sum() 

        self.critic_net.zero_grad()
        loss.backward()
        self.critic_optim.step()

        self.update_counter += 1
        if self.update_counter >= self.target_update_freq:
            self.critic_target.hard_update(self.critic_net)
            self.update_counter = 0
        
        return loss.item()



def play_episode(env: gym.Env, agent: Agent, replay_buffer: ReplayBuffer, train=True):
    done = False
    agent.reset()
    s = env.reset()
    episode_reward = 0
    iters = 0

    while not done and iters < 2000:
        a = agent.get_action(s)
        next_s, r, terminal, truncated, info = env.step(a)
        done = terminal

        if terminal:
            r = -200 # -200 reward for letting pole collapse
        
        if train:
            replay_buffer.push(
                np.array(s, dtype=np.float32),
                np.array([a], dtype=np.float32),
                np.array(next_s, dtype=np.float32),
                np.array([r], dtype=np.float32),
                np.array([terminal], dtype=np.float32),
            )
            if len(replay_buffer) >= CARTPOLE_CONFIG['batch_size']:
                batch = replay_buffer.sample(CARTPOLE_CONFIG['batch_size'])
                agent.learn(batch)

        s = next_s
        episode_reward += r
        iters += 1
    
    return episode_reward
        

def main(n_episodes=300):
    env = gym.make('CartPole-v1', new_step_api=True)
    agent = DQN(env, RBF_Transform(env), CARTPOLE_CONFIG["hidden_size"],
        CARTPOLE_CONFIG["learning_rate"], CARTPOLE_CONFIG["gamma"], CARTPOLE_CONFIG["batch_size"],
        CARTPOLE_CONFIG["target_update_freq"])
    replay_buffer = ReplayBuffer(CARTPOLE_CONFIG["buffer_capacity"])
    rewards = np.zeros(n_episodes)

    for i in tqdm(range(n_episodes)):
        agent.epsilon = (0.99 ** i)
        rewards[i] = play_episode(env, agent, replay_buffer)

        if (i + 1) % 10 == 0:
            tqdm.write(f"Avg reward last 10 episodes: {np.mean(rewards[i-9:i+1])}")
            tqdm.write(f"Agent epsilon: {agent.epsilon}")
        
    plt.plot(rewards)
    plt.show()
    plot_avg_reward(rewards)

    agent.epsilon = 0
    env = wrappers.RecordVideo(env, 'cartpole_dqn', new_step_api=True)
    play_episode(env, agent, replay_buffer, False)


if __name__ == "__main__":
    main()