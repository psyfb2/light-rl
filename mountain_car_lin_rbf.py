from re import S
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from gym import wrappers
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm

from agent import Agent
from plotting import plot_avg_reward


class QLearningLinearRBF(Agent):
    def __init__(self, env: gym.Env, epsilon=0.9, gamma=0.99, n=1):
        """
        Q learning agent, assumes action space is discrete and represented using
        integer from 0 to num_actions. Uses num_actions linear models and function approximation
        """
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma

        # n step variables
        self.n = n
        self.sar_list = []

        # state feature transformer
        X = [env.observation_space.sample() for _ in range(10000)]
        self.standard_scaler = StandardScaler()
        self.rbfs = FeatureUnion(
            [(f"rbf{g}", RBFSampler(gamma=g, n_components=500)) for g in (0.1, 0.5, 1, 2)]
        )
        self.rbfs.fit(self.standard_scaler.fit_transform(X))

        # init models (one per action)
        self.models = [SGDRegressor(learning_rate="constant") for _ in range(env.action_space.n)]
        for model in self.models:
            model.partial_fit(self._transform_state(env.reset()), [0])  # init model weights
        
    def reset(self):
        self.sar_list = []
    
    def _transform_state(self, s: np.ndarray) -> np.ndarray:
        return self.rbfs.transform(self.standard_scaler.transform([s]))
    
    def get_action(self, s: np.ndarray):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        s = self._transform_state(s)
        return np.argmax([model.predict(s)[0] for model in self.models])
    
    def learn(self, s: np.ndarray, a: int, r: float, next_s: np.ndarray, terminal: bool):
        s, next_s = self._transform_state(s), self._transform_state(next_s)

        self.sar_list.append((s, a, r))
        if len(self.sar_list) < self.n: return

        # update model for 0'th item in sar_list 
        g = sum(((self.gamma ** i) * self.sar_list[i][2] for i in range(len(self.sar_list))))

        if not terminal:
            g += (self.gamma ** self.n
                ) * np.max([model.predict(next_s)[0] for model in self.models])

            self.models[self.sar_list[0][1]].partial_fit(self.sar_list[0][0], [g])
            self.sar_list.pop(0)
        
        else:
            self.models[self.sar_list[0][1]].partial_fit(self.sar_list[0][0], [g])
            self.sar_list.pop(0)

            # finish of sar_list, all future rewards are 0 and going to stay on next_s
            for j in range(len(self.sar_list)):
                g = sum(((self.gamma ** i) * self.sar_list[i + j][2] for i in range(
                        len(self.sar_list) - j)))
                
                self.models[self.sar_list[j][1]].partial_fit(self.sar_list[j][0], [g])


def play_episode(env: gym.Env, agent: Agent):
    done = False
    s = env.reset()
    agent.reset()
    episode_reward = 0

    while not done:
        a = agent.get_action(s)
        next_s, r, terminal, truncated, info = env.step(a)
        done = terminal # or truncated

        agent.learn(s, a, r, next_s, terminal)

        s = next_s
        episode_reward += r
    
    return episode_reward


def plot_cost_to_go(env: gym.Env, agent: QLearningLinearRBF, num_tiles=30):
    # state is [position, velocity] with bounds
    # -1.2 <= pos <= 0.6, -0.07 <= vel <= 0.07
    # plot -V(s) for meshgrid of states
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)

    X, Y = np.meshgrid(x, y)  # shape => (num_tiles, num_tiles)
    Z = np.apply_along_axis(
        lambda _: -np.max([model.predict(agent._transform_state(_)) 
                           for model in agent.models]), 
        2, np.dstack([X, Y]))

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('-V(s)')
    ax.set_title("Cost to Go Function")
    plt.show()


def main(n_episodes=600):
    env = gym.make('MountainCar-v0', new_step_api=True)
    agent = QLearningLinearRBF(env, n=5)
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
    plot_cost_to_go(env, agent)

    env = wrappers.RecordVideo(env, 'mountaincar_qlearning_lin_rbf_vids', new_step_api=True)
    play_episode(env, agent)


if __name__ == "__main__":
    main()