import gym
import numpy as np

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

from light_rl.common.base.feature_transform import Transform

class RBFTransform(Transform):
    def __init__(self, env: gym.Env, n_episodes=1000, n_components=200, gammas=(0.05, 0.1, 0.5, 1.0)):
        """ RBF state feature transformer. Gather states using n_episodes rollouts of uniform random
        policy. Use states to fit standard scaler and RBFSamplers. This can then be used to transform
        future states. This is useful as a feature extractor for linear RL models. transformed state dim
        will be n_components*len(gammas)

        Args:
            env (gym.Env): Enviroment in which to play episodes. Must follow gym
                interface.
            n_episodes (int, optional): number of rollouts with uniform random polciy. 
                Defaults to 1000.
            n_components (int, optional): Dimensionality of rbf. Defaults to 200.
            gammas (tuple, optional): _description_. Defaults to (0.05, 0.1, 0.5).
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
        self.rbfs = FeatureUnion(
            [(f"rbf{g}", RBFSampler(gamma=g, n_components=n_components)) for g in gammas]
        )
        self.rbfs.fit(self.standard_scaler.fit_transform(states))
    
    def transform(self, s: np.ndarray) -> np.ndarray:
        """ Transform given state.

        Args:
            s (np.ndarray): state to transform. Can also be
                batch of states (shape => (batch_size, state_size))

        Returns:
            np.ndarray: new transformed state(s)
        """
        if s.ndim == 1: # 1d arr (sklearn expects 2d arr)
            return self.rbfs.transform(self.standard_scaler.transform([s]))[0]
        return self.rbfs.transform(self.standard_scaler.transform(s))
