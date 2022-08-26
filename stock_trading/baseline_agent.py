import numpy as np
import gym

from light_rl.common.base.agent import BaseAgent

class BaselineAgent(BaseAgent):
    def __init__(self, action_space: gym.Space, state_space: gym.Space,
            stock_price_indicies: np.ndarray, cash_idx: int, max_actions: int):
        """ Constructor for BaselineAgent. This agent works with StockEnv
        by buying stocks and holding. Assumes stock_price in the state
        is raw prices.

        Args:
            action_space (gym.Space): action space
            state_space (gym.Space): observation space 
            stock_price_indicies (np.ndarray): indicies of the raw stock prices
                within any received obvservation.
            cash_idx (int): index of cash within any received observation.
            max_actions (int): max_action value used with the StockEnv.
        """
        super().__init__(action_space, state_space)
        self.stock_price_indicies = stock_price_indicies
        self.cash_idx = cash_idx
        self.max_action = max_actions
    
    def get_action(self, state: np.ndarray, explore=False, rec_state=None):
        """ Buy stocks in a round robin until cash state is less than
        all stock prices.

        Args:
            state (np.ndarray): observation from StockEnv
            explore (bool, optional): does nothing. Defaults to False.
            rec_state (bool, optional): does nothing. Defaults to None.

        Returns:
            action (np.ndarray): corrosponding buy and hold action
            rec_state (Any): None
        """
        stock_prices = state[self.stock_price_indicies]
        cash = state[self.cash_idx]
        buy_actions = np.zeros_like(stock_prices)

        bought_stock = True
        while bought_stock:
            bought_stock = False
            for i in range(len(stock_prices)):
                if cash >= stock_prices[i]:
                    buy_actions[i] += 1
                    cash -= stock_prices[i]
                    bought_stock = True

        return buy_actions / self.max_action, None
    
    def train(self, env: gym.Env, max_timesteps: int, max_training_time: float, 
              target_return: float, max_episode_length: int, eval_freq: int, eval_episodes: int):
        return None, None
