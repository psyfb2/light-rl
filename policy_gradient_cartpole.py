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

from agent import Agent
from plotting import plot_avg_reward
from fc_network import FCNetwork, DEVICE


class ActorCritic(Agent):
    def __init__(self, env: gym.Env, actor_hidden_layers=[],
            critic_hidden_layers=[128], actor_lr=1e-3, critic_lr=1e-4, gamma=1.0):
        """
        Actor Critic agent, assumes action space is discrete and represented using
        integer from 0 to num_actions.
        """
        self.env = env
        self.gamma = gamma

        # state feature transformer
        states = []
        for _ in range(10000):
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
            [(f"rbf{g}", RBFSampler(gamma=g, n_components=1000)) for g in (0.05, 0.1, 0.5, 1)]
        )
        self.rbfs.fit(self.standard_scaler.fit_transform(states))

        # init models
        self.state_dim = self._transform_state(env.reset()).shape[-1]
        self.action_dim = env.action_space.n

        self.actor_net = FCNetwork(
            (self.state_dim, *actor_hidden_layers, self.action_dim), 
            lambda: torch.nn.modules.activation.Softmax(dim=-1)
        )
        self.actor_optim = Adam(
            self.actor_net.parameters(), lr=actor_lr, eps=1e-3
        )

        self.critic_net = FCNetwork(
            (self.state_dim, *critic_hidden_layers, 1)
        )
        self.critic_optim = Adam(
            self.critic_net.parameters(), lr=critic_lr, eps=1e-3
        )

        self.reset()
    
    def reset(self):
        self.gamma_t = 1
    
    def _transform_state(self, s: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            self.rbfs.transform(self.standard_scaler.transform([s]))[0]
        ).to(DEVICE).float()
    
    def get_action(self, s: np.ndarray) -> int:
        self.s = s
        with torch.no_grad():
            s = self._transform_state(s)

        self.pmf = Categorical(self.actor_net(s))
        self.a = self.pmf.sample()
        return self.a.item()
    
    def learn(self, s: np.ndarray, a: int, r: float, next_s: np.ndarray, terminal: bool):
        '''
        learn should be passed (s, a, r, next_s, terminal) for (s, a) from last call
        to get_action.
        '''
        # (s, a) needs to be from last call to get_action
        assert a == self.a
        assert (s == self.s).all() 


        with torch.no_grad():
            s, next_s = self._transform_state(s), self._transform_state(next_s)

            # calculate r + gamma * v(next_s)
            g = r
            if not terminal:
                g += self.gamma * self.critic_net(next_s).item()
            
        td_error = g - self.critic_net(s)

        # update critic using MSE(g, v(s))
        self.critic_net.zero_grad()
        loss = (td_error).pow(2).sum() 
        loss.backward()
        self.critic_optim.step()

        # update the actor using Policy Gradient Theorom
        self.actor_net.zero_grad()
        td_error = td_error.detach()  # so loss is not back-propagated to critic params
        loss = -( self.gamma_t *  td_error * self.pmf.log_prob(self.a))
        loss.backward()
        self.actor_optim.step()

        self.gamma_t *= self.gamma



def play_episode(env: gym.Env, agent: Agent):
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

        agent.learn(s, a, r, next_s, terminal)

        s = next_s
        episode_reward += r
        iters += 1
    
    return episode_reward
        

def main(n_episodes=1000):
    env = gym.make('CartPole-v1', new_step_api=True)
    agent = ActorCritic(env)
    rewards = np.zeros(n_episodes)

    for i in tqdm(range(n_episodes)):
        rewards[i] = play_episode(env, agent)

        if (i + 1) % 10 == 0:
            tqdm.write(f"Avg reward last 10 episodes: {np.mean(rewards[i-9:i+1])}")
        
    plt.plot(rewards)
    plt.show()
    plot_avg_reward(rewards)

    agent.epsilon = 0
    env = wrappers.RecordVideo(env, 'cartpole_policy_grad_vids', new_step_api=True)
    play_episode(env, agent)


if __name__ == "__main__":
    main()