import numpy as np

from stock_trading.baseline_agent import BaselineAgent
from stock_trading.stock_env import StockEnv
from light_rl.common.base.agent import BaseAgent


def play_episode(env: StockEnv, agent: BaseAgent):
    """ Play episode with trained agent. Once episode
    is finished will plot portfolio value over time.

    Args:
        env (StockEnv): StockEnv object
        agent (BaseAgent): agent to play episode with

    Returns:
        episode_reward (float): reward from the episode
        portfolio_val_hist (np.ndarray): np array of
            portfolio values with shape (num_trading_days, )
    """
    done = False
    agent.reset()
    s = env.reset()
    episode_reward = 0
    rec_state = None

    while not done:
        a, rec_state = agent.get_action(s, False, rec_state)
        next_s, r, done, _, info = env.step(a)

        s = next_s
        episode_reward += r
    
    env.render(block=True)
    return episode_reward, env.portfolio_val_hist.copy()


def run_baseline(env: StockEnv):
    """ Train baseline agent, then play an episode
    and return reward and portfolio values for that episode.
    Assumes env uses raw prices.

    Args:
        env (StockEnv): StockEnv. Must use raw prices.

    Returns:
        episode_reward (float): reward from the episode
        portfolio_val_hist (np.ndarray): np array of
            portfolio values with shape (num_trading_days, )
    """
    if not env.use_raw_prices:
        raise ValueError("Argument 'env' must use raw prices")

    agent = BaselineAgent(
        env.action_space, env.observation_space, 
        env.STOCK_PRICE_INDICIES, env.CASH_INDEX, env.max_action
    )

    return play_episode(env, agent)


if __name__ == "__main__":
    tests()
    # env = StockEnv(use_raw_prices=True)
    # run_baseline(env)