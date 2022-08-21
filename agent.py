from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pickle

class Agent(ABC):
    @abstractmethod
    def get_action(self, state: np.ndarray):
        pass

    def learn(self, s: np.ndarray, a: Union[np.ndarray, float, tuple, int],
              r: float, next_s: np.ndarray, next_s_terminal: bool):
        pass

    def reset(self):
        # call before episode starts, so that agent can restart any variables if needed
        pass

    def save(self, fn: str):
        with open(fn, 'wb') as handle:
            pickle.dump(self, handle)
    
    @staticmethod
    def load(fn: str):
        with open(fn, 'rb') as handle:
            return pickle.load(handle)
