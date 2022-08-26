import numpy as np

from stock_trading.baseline_agent import BaselineAgent
from stock_trading.stock_env import StockEnv


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
        tickers=None, start_date=None, end_date=None,
        time_period=None, open_prices=stock_prices,
        tech_indicators=None, init_investment=init_investment,
        max_action=max_action, use_raw_prices=True, new_step_api=False
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
        tickers=None, start_date=None, end_date=None,
        time_period=None, open_prices=stock_prices,
        tech_indicators=None, init_investment=init_investment,
        max_action=max_action, use_raw_prices=False, new_step_api=False
    )
    assert env.use_raw_prices is False
    assert (env.stock_prices == np.array([
        [5, 5, -5], # day2
        [0, 5, 15], # day3
        [-15, -15, 5]   # day4
    ])).all()

    stock_prices = np.array([
        [10, 15, 20], # day1
        [15, 20, 15], # day2
        [15, 25, 30], # day3
        [0, 10, 35]   # day4
    ])
    init_investment = 50
    env = StockEnv(
        tickers=None, start_date=None, end_date=None,
        time_period=None, open_prices=stock_prices,
        tech_indicators=None, init_investment=init_investment,
        max_action=max_action, use_raw_prices=True, new_step_api=False
    )

    state = env.reset()
    assert (state == np.array([10, 15, 20, 0, 0, 0, 50])).all()
    assert (env.STOCK_PRICE_INDICIES == np.array([0, 1, 2])).all()
    assert (env.SHARES_OWNED_INDICIES == np.array([3, 4, 5])).all()
    assert env.CASH_INDEX == 6
    assert env.n_stocks == 3


    stock_prices = np.array([
        [10, 15, 20], # day1
        [15, 20, 15], # day2
        [15, 25, 30], # day3
        [0, 10, 35]   # day4
    ])
    num_indicators = 6
    tech_indicators = np.tile(np.arange(num_indicators), (stock_prices.shape[0], 1))
    env = StockEnv(
        tickers=None, start_date=None, end_date=None,
        time_period=None, open_prices=stock_prices,
        tech_indicators=tech_indicators, init_investment=init_investment,
        max_action=max_action, use_raw_prices=True, new_step_api=False
    )

    state = env.reset()
    assert (state == np.array([10, 15, 20, 0, 0, 0, 50] + list(range(num_indicators)))).all()
    assert (env.STOCK_PRICE_INDICIES == np.array([0, 1, 2])).all()
    assert (env.SHARES_OWNED_INDICIES == np.array([3, 4, 5])).all()
    assert env.CASH_INDEX == 6
    assert env.n_stocks == 3

    next_s, r, done, info = env.step([0.02, 0.02, -0.01]) # day1 action
    assert (next_s == np.array([15, 20, 15, 2, 2, 0, 0] + list(range(num_indicators)))).all()
    assert r == 20
    assert done is False
    assert info["portfolio_value"] == 70

    next_s, r, done, info = env.step([0, -0.02, 0.02]) # day2 action
    assert (next_s == np.array([15, 25, 30, 2, 0, 2, 10] + list(range(num_indicators)))).all()
    assert r == 30
    assert done is False
    assert info["portfolio_value"] == 100

    next_s, r, done, info = env.step([0, 0, 0]) # day3 action
    assert (next_s == np.array([0, 10, 35, 2, 0, 2, 10] + list(range(num_indicators)))).all()
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
    num_indicators = 6
    tech_indicators = np.tile(np.arange(num_indicators), (stock_prices.shape[0], 1))
    env = StockEnv(
        tickers=None, start_date=None, end_date=None,
        time_period=None, open_prices=stock_prices,
        tech_indicators=tech_indicators, init_investment=init_investment,
        max_action=max_action, use_raw_prices=False, new_step_api=False
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
    assert env.n_stocks == 3
    assert env.stock_prices.shape[0] == env.tech_indicators.shape[0]
    assert env.stock_prices.ndim == env.tech_indicators.ndim == 2
    raw_prices = env.stock_prices

    env = StockEnv(
        tickers=["AAPL", "MSI", "SBUX"], start_date="2022-01-01", 
        end_date="2022-01-08", use_raw_prices=False
    )
    assert env.n_stocks == 3
    assert env.stock_prices.shape[0] == env.tech_indicators.shape[0]
    assert env.stock_prices.ndim == env.tech_indicators.ndim == 2
    assert np.isclose(env.stock_prices[1:], np.diff(raw_prices, axis=0)).all()