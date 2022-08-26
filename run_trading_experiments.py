import numpy as np

from stock_trading.baseline_agent import BaselineAgent
from stock_trading.stock_env import StockEnv
from light_rl.common.base.agent import BaseAgent


def tests():
    # TODO: package this into unit tests
    max_action = 100
    agent = BaselineAgent(
        None, None, np.array([0, 1, 2]), -1, max_action
    )

    a, _ = agent.get_action(np.array([10, 15, 20, 0, 0, 0, 50]))
    assert (a == np.array([1, 1, 1]) / max_action).all()

    a, _ = agent.get_action(np.array([15, 20, 15, 2, 2, 0, 0]))
    assert (a == np.array([0, 0, 0]) / max_action).all()

    
    a, _ = agent.get_action(np.array([15, 25, 30, 2, 0, 2, 10]))
    assert (a == np.array([0, 0, 0]) / max_action).all()

    a, _ = agent.get_action(np.array([15, 25, 30, 2, 0, 2, 200.5]))
    assert (a == np.array([4, 3, 2]) / max_action).all()

    stock_prices = np.array([
        [10, 15, 20], # day1
        [15, 20, 15], # day2
        [15, 25, 30], # day3
        [0, 10, 35]   # day4
    ])
    init_investment = 50
    env = StockEnv(
        None, None, None, stock_prices, init_investment, 
        100, True, new_step_api=False
    )

    state = env.reset()
    assert (state == np.array([10, 15, 20, 0, 0, 0, 50])).all()
    assert (env.STOCK_PRICE_INDICIES == np.array([0, 1, 2])).all()
    assert (env.SHARES_OWNED_INDICIES == np.array([3, 4, 5])).all()
    assert env.CASH_INDEX == 6
    assert env.n_stocks == 3

    next_s, r, done, info = env.step([0.02, 0.02, -0.01]) # day1 action
    assert (next_s == np.array([15, 20, 15, 2, 2, 0, 0])).all()
    assert r == 20
    assert done is False
    assert info["portfolio_value"] == 70

    next_s, r, done, info = env.step([0, -0.02, 0.02]) # day2 action
    assert (next_s == np.array([15, 25, 30, 2, 0, 2, 10])).all()
    assert r == 30
    assert done is False
    assert info["portfolio_value"] == 100

    next_s, r, done, info = env.step([0, 0, 0]) # day3 action
    assert (next_s == np.array([0, 10, 35, 2, 0, 2, 10])).all()
    assert r == -20
    assert done is True
    assert info["portfolio_value"] == 80

    assert (env.portfolio_val_hist == np.array([50, 70, 100, 80])).all()

    stock_prices = np.array([
        [10, 15, 20], # day1
        [15, 20, 15], # day2
        [15, 25, 30], # day3
        [0, 10, 35]   # day4
    ])
    assert (env.stock_prices == stock_prices).all() # env should not modify stock prices
    env = StockEnv(
        None, None, None, stock_prices, init_investment, 
        100, False, new_step_api=False
    )
    assert env.use_raw_prices is False
    assert (env.stock_prices == np.array([
        [5, 5, -5], # day2
        [0, 5, 15], # day3
        [-15, -15, 5]   # day4
    ])).all()

    # requires internet connection (TODO: mock the yfinance api during testing)
    env = StockEnv(
        tickers=["AAPL", "MSI", "SBUX"], start_date="2022-01-01", 
        end_date="2022-01-08", use_raw_prices=True
    )
    assert env.stock_prices.shape == (5, 3)
    assert env.n_stocks == 3


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
    env = StockEnv(use_raw_prices=True)
    run_baseline(env)