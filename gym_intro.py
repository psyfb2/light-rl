import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from tqdm import tqdm

from agent import Agent
from incremental_avg import IncrementalAvg

all_states = []

class TwoActionLinearAgent(Agent):
    def __init__(self, weight_dims: tuple):
        self.weight_dims = weight_dims
        self.reset()
    
    def reset(self):
        self.weights = np.random.randn(*self.weight_dims)
    
    def get_action(self, s):
        if np.dot(self.weights, s) > 0:
            return 0
        return 1


def play_episode(env: gym.Env, agent: Agent):
    done = False
    s = env.reset()
    episode_reward = 0

    while not done:
        if len(all_states) < 1000:
            all_states.append(s)
        a = agent.get_action(s)

        next_s, r, done, truncated, info = env.step(a)

        done = done or truncated

        s = next_s
        episode_reward += r
    
    return episode_reward
        

def main(n_averages=1000, n_episodes=25):
    env = gym.make('CartPole-v1', new_step_api=True)
    agent = TwoActionLinearAgent((4, ))
    best_avg, best_weights = float("-inf"), None
    avg_r = IncrementalAvg()

    for i in tqdm(range(n_averages)):
        avg_r.reset()
        agent.reset()

        for j in range(n_episodes):
            avg_r.update(play_episode(env, agent))
        
        if avg_r.avg > best_avg:
            best_avg = avg_r.avg
            best_weights = agent.weights
    
    agent.weights = best_weights
    print("Best avg reward:", best_avg)
    print("Best weights:", best_weights)

    env = wrappers.RecordVideo(env, 'cartpole_random_search_vids', new_step_api=True)
    play_episode(env, agent)

    plt.hist(all_states)
    plt.show()





if __name__ == "__main__":
    main()