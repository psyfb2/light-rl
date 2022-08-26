import numpy as np
import gym
import yfinance as yf
import matplotlib.pyplot as plt

from typing import List


class StockEnv(gym.Env):
    metadata = {'render.modes': ['live']}

    def __init__(self, tickers=["AAPL", "MSI", "SBUX", "GOOGL", "GOLD"], 
            start_date="2017-01-01", end_date="2021-01-01",
            open_prices: np.ndarray = None, init_investment=int(2e4), 
            max_action=100, use_raw_prices=False, new_step_api=True):
        """ Constructor for StockEnv.

        Args:
            tickers (list, optional): stocks to use for this enviroment.
                 Defaults to ["AAPL", "MSI", "SBUX", "GOOGL", "GOLD"].
            start_date (str, optional): start date as yyyy-mm-dd str. 
                Defaults to "2017-01-01".
            end_date (str, optional): end date as yyyy-mm-dd str.
                Defaults to "2021-01-01".
            open_prices (np.ndarray, optional): if this is not None
                then tickers, start_date and end_date are ignored, will
                use this as the stock open price data,
                must be shape (num_stocks, num_trading_days). Defaults to None.
            init_investment (int, optional): cash at start of trading. 
                Defaults to int(2e4).
            max_action (int, optional): The maximum number of shares
                that can be bought or sold in one trading day per share. Defaults to 100.
            use_raw_prices (bool, optional): If True the state space will use
                raw stock prices. However this is not usually optimal for
                ML models since regression does not do well during extrapolation
                and the general trend for stocks is to go up, so your agent may
                end up extrapolating when it receives states which were not
                close to any states seen in training, therefore meaning
                it would extrapolate. If False would use the difference in
                price (i.e. the state of the price for a stock would be
                todays price - yesterdays price), this differencing
                can make models resilliant to non-stationary price data.
                Note if False, will drop data for first trading day.

            new_step_api (bool, optional): Whether or not to use
                the openai gym two_step_api. Defaults to True.
            
        """
        super().__init__()
        if open_prices is not None:
            if open_prices.ndim != 2:
                raise ValueError(
                    f"Argument 'open_prices' must have shape (num_trading_days, num_stocks), "
                    f"but received shape {open_prices.shape}"
                )
            self.stock_prices = open_prices
            self.n_stocks = open_prices.shape[1]
            self.tickers = self.start_date = self.end_date = None
        else:
            self.stock_prices = self._get_historical_share_prices(tickers, start_date, end_date)
            self.n_stocks = len(tickers)
            self.tickers = tickers
            self.start_date = start_date
            self.end_date = end_date
    
        if not use_raw_prices:
            # val(t) = price(t) - price(t - 1), will lose first trading day
            self.stock_prices = np.diff(self.stock_prices, axis=0)

        self.init_investment = init_investment
        self.max_action = max_action
        self.use_raw_prices = use_raw_prices
        self.new_step_api = new_step_api

        self.ptr = 0
        self.portfolio_val = self.cash_state = self.stock_price_state = self.shares_owned_state = None
        self.portfolio_val_hist = np.zeros((self.stock_prices.shape[0], ))

        self.STOCK_PRICE_INDICIES = np.array([i for i in range(self.n_stocks)])
        self.SHARES_OWNED_INDICIES = np.array([self.n_stocks + i for i in range(self.n_stocks)])
        self.CASH_INDEX = 2*self.n_stocks

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.n_stocks, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.NINF, high=np.inf, shape=(2*self.n_stocks+1, ), dtype=np.float32
        )

    @staticmethod
    def _get_historical_share_prices(tickers: List[str], start: str, end: str) -> np.ndarray:
        """ Get historical share price data

        Args:
            tickers (List[str]): list of tickers
            start (str): start date as yyyy-mm-dd str
            end (str): end date as yyyy-mm-dd str

        Returns:
            np.ndarray: open prices in shape (trading_days, len(tickers))
        """
        data = yf.download(tickers, start=start, end=end)
        data = data[[('Open', tkr) for tkr in tickers]]
        return data.to_numpy()

    def reset(self) -> np.ndarray:
        """ Reset the enviroment by going back to 
        the first trading day.

        Returns:
            np.ndarray: initial state. First n_stocks values are the
                stock prices. Second n_stocks values are the shares
                owned, element after is balance. Then elements after
                are technical indicators.
        """
        self.ptr = 0
        self.portfolio_val = self.cash_state = self.init_investment
        self.stock_price_state = self.stock_prices[0]
        self.shares_owned_state = np.array([0 for _ in self.stock_price_state])
        self.portfolio_val_hist[0] = self.cash_state
        return np.concatenate((self.stock_price_state, self.shares_owned_state, [self.cash_state]))
    
    def step(self, a: np.ndarray):
        """ Step to perform trade today and move to next trading day. Assumes
        trading happens at the start of the day using open prices (i.e. 
        will buy and sell stocks at todays open price). Also assumes
        funds from sales are available immediately, so will perform
        all sell actions first. 

        Args:
            a (np.ndarray): shape (num_stocks, ) where each element in [-1, 1]
                each element is then multiplied by self.max_action, so each
                element in a would now be in [-self.max_action, self.max_action]
                and rounded to the nearest integer.
                Each element specifies number of stocks to buy for corrosponding stock.

        Returns:
            observation (np.ndarray): state for next trading day
            reward (float): reward for trade performed today. Which is 
                change in portfolio value
            terminal (bool): whether observation is a terminal state
            truncated (bool): whether this episode should be prematurly ended
                (should always be False)
            info (dict): info dict for debugging
        """
        if isinstance(a, np.ndarray): a = a.copy()
        else: a = np.array(a)

        if a.shape != (self.n_stocks,):
            raise ValueError(f"Argument 'a' should have shape ({self.n_stocks},) not {a.shape}")
        for i in range(len(a)):
            if a[i] > 1 or a[i] < -1:
                raise ValueError(f"{i}'th element of Argument 'a' is {a[i]}, all elements should be between -1 and 1.")
            a[i] = round(a[i] * self.max_action)
        
        # perform trades
        # 1. sell shares for sell actions
        for i in range(len(a)):
            if a[i] < 0:
                # n_sales = minimum(stocks to sell, stocks owned)
                n_sales = min(-a[i], self.shares_owned_state[i])
                self.cash_state += n_sales * self.stock_price_state[i]
                self.shares_owned_state[i] -= n_sales

        # 2. buy shares for buy actions
        for i in range(len(a)):
            if a[i] > 0:
                # n_buys = minimum(max num stocks that can be bought with cash, stocks to buy)
                n_buys = min(self.cash_state // self.stock_price_state[i], a[i])
                self.cash_state -= n_buys * self.stock_price_state[i]
                self.shares_owned_state[i] += n_buys

        self.ptr += 1  # the next trading day
        done = self.ptr >= len(self.stock_prices) - 1
        self.stock_price_state = self.stock_prices[self.ptr]

        # calculate reward, next state, portfolio value, done
        portfolio_val = np.dot(self.stock_price_state, self.shares_owned_state) + self.cash_state
        reward = portfolio_val - self.portfolio_val  # change in portfolio value
        self.portfolio_val = portfolio_val
        self.portfolio_val_hist[self.ptr] = portfolio_val

        next_state = np.concatenate((self.stock_price_state, self.shares_owned_state, [self.cash_state]))

        if self.new_step_api:
            return next_state, reward, done, False, {"portfolio_value": self.portfolio_val}
        return next_state, reward, done, {"portfolio_value": self.portfolio_val}
    
    def render(self, mode='live', title="Portfolio Value", block=False):
        """ Plot the portfolio value up to the current trading day.

        Args:
            mode (str, optional): render mode. Defaults to 'live'.
            title (str, optional): title of plot. Defaults to "Portfolio Value".
            block (bool, optional): whether or not to block the program 
                untill plot is closed. Defaults to False.
        """
        fig = plt.figure()
        fig.suptitle(title)
        plt.plot(self.portfolio_val_hist)
        plt.xlabel("Trading Day")
        plt.ylabel("Portfolio Value")
        plt.show(block=block)

    



