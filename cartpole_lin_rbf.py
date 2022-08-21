import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from gym import wrappers
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm

from agent import Agent
from plotting import plot_avg_reward
from linear_regression import LinearRegression


class QLearningLinearRBF(Agent):
    def __init__(self, env: gym.Env, epsilon=0.9, gamma=0.99):
        """
        Q learning agent, assumes action space is discrete and represented using
        integer from 0 to num_actions. Uses num_actions linear models and function approximation
        """
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma

        # state feature transformer
        states = []
        for _ in range(5000):
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

        # init models (one per action)
        self.models = [LinearRegression(self._transform_state(env.reset()).shape[-1], lr=10e-2) 
                       for _ in range(env.action_space.n)]
    
    def _transform_state(self, s: np.ndarray) -> np.ndarray:
        return self.rbfs.transform(self.standard_scaler.transform([s]))
    
    def get_action(self, s: np.ndarray):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        s = self._transform_state(s)
        return np.argmax([model.predict(s)[0] for model in self.models])
    
    def learn(self, s: np.ndarray, a: int, r: float, next_s: np.ndarray, terminal: bool):
        s, next_s = self._transform_state(s), self._transform_state(next_s)

        g = r
        if not terminal: 
            g += self.gamma * np.max([model.predict(next_s)[0] for model in self.models])
        
        self.models[a].partial_fit(s, np.array([g]))


def play_episode(env: gym.Env, agent: Agent):
    done = False
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
        

def main(n_episodes=600):
    env = gym.make('CartPole-v1', new_step_api=True)
    agent = QLearningLinearRBF(env)
    rewards = np.zeros(n_episodes)

    for i in tqdm(range(n_episodes)):
        agent.epsilon = 0.1 * (0.997 ** i)
        rewards[i] = play_episode(env, agent)

        if (i + 1) % 100 == 0:
            tqdm.write(f"Avg reward last 100 episodes: {np.mean(rewards[i-99:i+1])}")
            tqdm.write(f"Epsilon: {agent.epsilon}")
        
    plt.plot(rewards)
    plt.show()
    plot_avg_reward(rewards)

    agent.epsilon = 0
    env = wrappers.RecordVideo(env, 'cartpole_qlearning_lin_rbf_vids', new_step_api=True)
    play_episode(env, agent)


if __name__ == "__main__":
    main()