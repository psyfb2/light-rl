import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from tqdm import tqdm
from collections import defaultdict

from agent import Agent
from incremental_avg import IncrementalAvg


# turns list of integers into an int
# Ex.
# build_state([1,2,3,4,5]) -> 12345
def build_state(features):
  return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
  return np.digitize(x=[value], bins=bins)[0]


class QLearningBinAgent(Agent):
    def __init__(self, random_act_func, epsilon=0.9, lr=0.1, gamma=0.9, agent_learns=True, 
                 decay_func=lambda eps, lr : (eps, max(10e-3, lr*0.999))):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.random_act_func = self._random_act_func_wrapper(random_act_func)
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.decay_func = decay_func
        self.agent_learns = agent_learns

        # Note: to make this better you could look at how often each bin was
        # actually used while running the script.
        # It's not clear from the high/low values nor sample() what values
        # we really expect to get.
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9) # (-inf, inf) (I did not check that these were good values)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9) # (-inf, inf) (I did not check that these were good values)
    
    def _random_act_func_wrapper(self, random_act_func):
        def wrapped(*args, **kwargs):
            a = random_act_func(*args, **kwargs)
            if isinstance(a, (list, np.ndarray)):
                a = tuple(a)
            return a
        return wrapped
    
    def _discritize_state(self, state: np.ndarray) -> np.ndarray:
        cart_pos, cart_vel, pole_angle, pole_vel = state
        return build_state([
            to_bin(cart_pos, self.cart_position_bins),
            to_bin(cart_vel, self.cart_velocity_bins),
            to_bin(pole_angle, self.pole_angle_bins),
            to_bin(pole_vel, self.pole_velocity_bins),
        ])

    def get_action(self, state: np.ndarray) -> tuple:
        disc_state = self._discritize_state(state)

        if self.agent_learns and np.random.rand() < self.epsilon:
            a = self.random_act_func(state)
            self.q_table[disc_state][a] = 0.0  # initialise q-table value to 0
            return a
        else:
            # argmax q(s, a) over all actions a
            if len(self.q_table[disc_state].items()) == 0:
                # empty q-table, pick random action
                a = self.random_act_func(state)
                self.q_table[disc_state][a] = 0.0  # initialise q-table value to 0
                return a
            
            max_a, max_val = None, float("-inf")
            for a, val in self.q_table[disc_state].items():
                if val > max_val:
                    max_val = val
                    max_a = a

            return max_a
    
    def learn(self, s: np.ndarray, a: np.ndarray, r: float, next_s: np.ndarray, d: bool):
        if not self.agent_learns:
            return
        
        if isinstance(a, (list, np.ndarray)):
            a = tuple(a)

        s, next_s = self._discritize_state(s), self._discritize_state(next_s)

        g = r
        if not d:
            g += self.gamma * max(tuple(val for _, val in self.q_table[next_s].items()) + (0, ))

        self.q_table[s][a] = self.q_table[s][a] + self.lr * (g - self.q_table[s][a])
        self.epsilon, self.lr = self.decay_func(self.epsilon, self.lr)


def play_episode(env: gym.Env, agent: Agent):
    done = False
    s = env.reset()
    episode_reward = 0
    iters = 0

    while not done:
        a = agent.get_action(s)
        next_s, r, done, truncated, info = env.step(a)
        episode_reward += r

        if done and iters < 199:
            r = -300

        agent.learn(s, a, r, next_s, done and not truncated)

        s = next_s
        iters += 1
    
    return episode_reward
        

def main(n_episodes=10000):
    env = gym.make('CartPole-v1', new_step_api=True)
    agent = QLearningBinAgent(lambda s: env.action_space.sample())
    avg_r = IncrementalAvg()
    avg_r_arr = []

    for i in tqdm(range(n_episodes)):
        agent.epsilon = 1.0 / np.sqrt(i + 1)
        avg_r.update(play_episode(env, agent))

        if i % 100 == 0:
            tqdm.write(f"avg episode reward last 1000 episodes: {round(avg_r.avg, 4)}")
            tqdm.write(f"eps: {round(agent.epsilon, 4)}, lr: {round(agent.lr, 4)}")
            avg_r_arr.append(avg_r.avg)
            avg_r.reset()

    plt.plot(avg_r_arr)
    plt.show()

    env = wrappers.RecordVideo(env, 'cartpole_qlearningbins_vids', new_step_api=True)
    play_episode(env, agent)



if __name__ == "__main__":
    main()