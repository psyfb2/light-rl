import gym
import numpy as np

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from light_rl.common.base.feature_transform import Transform

class StandardScalerTransform(Transform):
    def __init__(self, env: gym.Env, n_episodes=1000):
        """ State feature transformer. Gather states using n_episodes rollouts of uniform random
        policy. Use states to fit standard scaler. This can then be used to transform
        future states. This is useful is elements in state are on vastly different scale.

        Args:
            env (gym.Env): Enviroment in which to play episodes. Must follow gym
                interface.
            n_episodes (int, optional): number of rollouts with uniform random polciy. 
                Defaults to 1000.
        """
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
        self.standard_scaler.fit(states)
    
    def transform(self, s: np.ndarray) -> np.ndarray:
        if len(s.shape) == 1: # 1d arr (sklearn expects 2d arr)
            return self.standard_scaler.transform([s])[0]
        return self.standard_scaler.transform(s)
