import numpy as np
import gym
import talib
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from typing import List


class StockEnv(gym.Env):
    metadata = {'render.modes': ['live']}

    def __init__(self, tickers=["AAPL", "MSI", "GOOGL", "SBUX", "GOLD", "BHP", "IBDRY"], 
            start_date="2010-01-01", end_date="2021-01-01", time_period=30,
            open_prices: np.ndarray = None, tech_indicators: np.ndarray = None,
            init_investment=int(2e4), max_action=100, 
            use_raw_prices=False, new_step_api=True):
        """ Constructor for StockEnv.

        Args:
            tickers (list, optional): stocks to use for this enviroment.
                Technical indicators will also be used when tickers are
                specified.
                Defaults to ["AAPL", "MSI", "SBUX", "GOOGL", "GOLD"].
            start_date (str, optional): start date as yyyy-mm-dd str.
                Defaults to "2017-01-01".
            end_date (str, optional): end date as yyyy-mm-dd str.
                Defaults to "2021-01-01".
            time_period (int, optional): time frame used for technical
                indicators
            open_prices (np.ndarray, optional): if this is not None
                then tickers, start_date, end_date and time_period are ignored, 
                will use this as the stock open price data,
                must be shape (num_trading_days, num_stocks). 
                Defaults to None.
            tech_indicators (np.ndarray, optional): Use in conjuction with
                open_price. Must have shape (num_trading_days, num_indicators). If
                None and open_prices is not None, then won't use any technical indicators.
                Defaults to None.
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
            if tech_indicators is not None and tech_indicators.ndim != 2:
                raise ValueError(
                    f"Argument 'tech_indicators' must have shape (num_trading_days, num_indicators), "
                    f"but received shape {open_prices.shape}"
                )
            if tech_indicators is not None and tech_indicators.shape[0] != open_prices.shape[0]:
                raise ValueError(
                    f"Argument 'tech_indicators' must have {open_prices.shape[0]} rows to match open_price, "
                    f"but received shape {open_prices.shape}"
                )

            self.stock_prices = open_prices
            self.tech_indicators = tech_indicators
            self.n_stocks = open_prices.shape[1]
            self.tickers = self.start_date = self.end_date = None
            if not use_raw_prices:
                # v(t) = price(t) - price(t - 1)
                # first day is lost when not using raw prices
                self.stock_prices = np.diff(self.stock_prices, axis=0)
                if self.tech_indicators is not None: self.tech_indicators = self.tech_indicators[1:, :]
        else:
            self.stock_prices, self.tech_indicators = self._get_historical_share_prices(
                tickers, start_date, end_date, time_period, use_raw_prices
            )
            self.n_stocks = len(tickers)
            self.tickers = tickers
            self.start_date = start_date
            self.end_date = end_date

        self.init_investment = init_investment
        self.max_action = max_action
        self.use_raw_prices = use_raw_prices
        self.new_step_api = new_step_api

        self.ptr = 0
        self.portfolio_val = self.cash_state = self.stock_price_state = self.shares_owned_state = None
        self.tech_state = None
        self.portfolio_val_hist = np.zeros((self.stock_prices.shape[0], ))

        self.STOCK_PRICE_INDICIES = np.array([i for i in range(self.n_stocks)])
        self.SHARES_OWNED_INDICIES = np.array([self.n_stocks + i for i in range(self.n_stocks)])
        self.CASH_INDEX = 2*self.n_stocks

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.n_stocks, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.NINF, high=np.inf, shape=(2*self.n_stocks+1, ), dtype=np.float32
        )

    @staticmethod
    def _get_historical_share_prices(tickers: List[str], start: str, end: str, 
            time_period=30, raw_prices=False) -> np.ndarray:
        """ Get historical share price data

        Args:
            tickers (List[str]): list of tickers
            start (str): start date as yyyy-mm-dd str
            end (str): end date as yyyy-mm-dd str

        Returns:
            open_prices (np.ndarray): open prices in shape (trading_days, len(tickers))
            technical_indicators (np.ndarray): indicators in shape (trading_days, num_indicators*num_stocks)
        """
        if time_period < 2:
            raise ValueError(f"Argument 'time_period' must be atleast 2, not {time_period}")
        
        # max(33, time_period) + 1 trading days removed from start 
        # due to technical indicators producing NaN (they need history), offset this
        start_offset = max(33, time_period) + 1
        if not raw_prices:
            # v(t) = price(t) - price(t - 1), would lose first trading day, offset this
            start_offset += 1
        start = (pd.to_datetime(start) - pd.offsets.BDay(start_offset)).strftime('%Y-%m-%d')

        data = yf.download(tickers, start=start, end=end)
        open_price = data[[('Open', tkr) for tkr in tickers]].to_numpy()  # => shape (n, num_stocks)

        indicator_funcs = [
            lambda i : talib.ADX(
                data[('High', tickers[i])], data[('Low', tickers[i])], 
                data[('Close', tickers[i])], timeperiod=time_period//2
            ), # shape => (trading_days, ), first time_period-1 elements are NaN

            lambda i : talib.MACD(
                data[('Close', tickers[i])], fastperiod=12, 
                slowperiod=26, signalperiod=9
            )[0], # shape => (trading_days, ), first 33 elements are NaN

            lambda i : talib.MACD(
                data[('Close', tickers[i])], fastperiod=12, 
                slowperiod=26, signalperiod=9
            )[1], # shape => (trading_days, ), first 33 elements are NaN

            lambda i : talib.RSI(
                data[('Close', tickers[i])], timeperiod=time_period
            ), # shape => (trading_days, ), first time_period elements are NaN

            lambda i : talib.BBANDS(
                data[('Close', tickers[i])], nbdevup=2, nbdevdn=2, 
                timeperiod=time_period
            )[0], # shape => (trading_days, ), first time_period-1 elements are NaN

            lambda i : talib.BBANDS(
                data[('Close', tickers[i])], nbdevup=2, nbdevdn=2, 
                timeperiod=time_period
            )[1], # shape => (trading_days, ), first time_period-1 elements are NaN

            lambda i : talib.BBANDS(
                data[('Close', tickers[i])], nbdevup=2, nbdevdn=2, 
                timeperiod=time_period
            )[2], # shape => (trading_days, ), first time_period-1 elements are NaN

            lambda i : talib.CCI(
                data[('High', tickers[i])], data[('Low', tickers[i])], 
                data[('Close', tickers[i])], timeperiod=time_period
            ), # shape => (trading_days, ), first time_period-1 elements are NaN

            lambda i : talib.OBV(
                data[('Close', tickers[i])], data[('Volume', tickers[i])]
            ) # shape => (trading_days, ), no NaN elements
        ]

        # shape => (num_indicators, num_stocks, trading_days)
        tech_indicators = np.zeros((len(indicator_funcs), len(tickers), data.shape[0]))
        for f_idx in range(len(indicator_funcs)):
            for i in range(len(tickers)):
                tech_indicators[f_idx, i, :] = indicator_funcs[f_idx](i)
        
        # the agent is assumed to trade in the morning, but these indicators use
        # close price. So day t needs to use indicators from day t - 1.
        # therefore tech indicators of last day not used
        tech_indicators = tech_indicators[:, :, :-1]
        open_price = open_price[1:, :]

        # reshape tech_indicators to (trading_days, num_indicators*num_stocks)
        # from (num_indicators, num_stocks, trading_days)
        tech_indicators = tech_indicators.swapaxes(0, -1)
        tech_indicators = tech_indicators.reshape(tech_indicators.shape[0], -1)

        # remove trading days that have NaNs in the technical indicators
        keep_arr = np.array(
            [not arr.any() for arr in np.isnan(tech_indicators)]
        )
        tech_indicators = tech_indicators[keep_arr]
        open_price = open_price[keep_arr]

        if not raw_prices:
            # v(t) = price(t) - price(t - 1), lost first element
            open_price = np.diff(open_price, axis=0)
            tech_indicators = tech_indicators[1:, :]

        return open_price, tech_indicators
        
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
        self.tech_state = self.tech_indicators[0] if self.tech_indicators is not None else []
        self.portfolio_val_hist[0] = self.cash_state
        return np.concatenate(
            (self.stock_price_state, self.shares_owned_state, [self.cash_state], self.tech_state)
        )
    
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
            info (dict): info dict containing portfolio value
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
        self.tech_state = self.tech_indicators[self.ptr] if self.tech_indicators is not None else []

        # calculate reward, next state, portfolio value, done
        portfolio_val = np.dot(self.stock_price_state, self.shares_owned_state) + self.cash_state
        reward = portfolio_val - self.portfolio_val  # change in portfolio value
        self.portfolio_val = portfolio_val
        self.portfolio_val_hist[self.ptr] = portfolio_val

        next_state = np.concatenate(
            (self.stock_price_state, self.shares_owned_state, [self.cash_state], self.tech_state)
        )

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
