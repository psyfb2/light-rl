import numpy as np
import pickle

from stock_trading.baseline_agent import BaselineAgent
from stock_trading.stock_env import StockEnv

from light_rl.common.base.agent import BaseAgent
from light_rl.algorithms.a3c import A3C
from light_rl.algorithms.ddpg import DDPG
from light_rl.algorithms.es import ES
from light_rl.algorithms.vanilla_policy_gradient import VanillaPolicyGradient
from light_rl.common.transforms.rbf import RBFTransform
from light_rl.common.transforms.standard_scaler import StandardScaler


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
    
    return episode_reward, env.portfolio_val_hist.copy()


def run_baseline(env: StockEnv):
    """ play an episode with baseline agent
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


def run_ddpg(train_env: StockEnv, extended_train_env: StockEnv, 
             val_env: StockEnv, test_env: StockEnv, 
             train_timelimit=60*60):
    """ Train DDPG. Tune hyperparams on val env.
    Train on extended_train_env with best hyperparam setting
    and then eval on test_env. Return reward and portfolio values 
    for single episode on test_env.

    Args:
        train_env (StockEnv): training env
        extended_train_env (StockEnv): training env for best hyperparam setting
            (included validation data)
        val_env (StockEnv): validation env
        test_env (StockEnv): test env

    Returns:
        episode_reward (float): reward from the episode
        portfolio_val_hist (np.ndarray): np array of
            portfolio values with shape (num_trading_days, )
    """
    # try most pluasible hyperparams
    train_episode_length = train_env.stock_prices.shape[0]
    train_episodes = 10000

    constant_config = {
        "max_timesteps": train_episodes * train_episode_length,
        "max_training_time": train_timelimit,
        "target_return": float("inf"),
        "max_episode_length": float("inf"),
        "eval_freq": 100 * train_episode_length,
        "eval_episodes": 1,
    }
    hyperparam_configs = [
        {
            "gamma": 0.99,
            "actor_learning_rate": 1e-3,
            "critic_learning_rate": 1e-3,
            "actor_adam_eps": 1e-3,
            "critic_adam_eps": 1e-3,
            "critic_hidden_layers": [64, 64],
            "actor_hidden_layers": [64, 64],
            "noise_std": 0.01,
            "tau": 0.01,
            "max_grad_norm": 50,
            "batch_size": 64,
            "buffer_capacity": int(1e6),
        }
    ]

    return play_episode(env, agent)



def load_envs():
    TRAIN_ENV_FN          = "train_env.pkl"
    EXTENDED_TRAIN_ENV_FN = "extended_train_env.pkl"
    VAL_ENV_FN            = "val_env.pkl"
    TEST_ENV_FN           = "test_env.pkl"
    TEST_ENV_RAW_FN       = "test_env_raw.pkl"

    try:
        with open(TRAIN_ENV_FN, 'rb') as handle:
            train_env = pickle.load(handle)
        
        with open(EXTENDED_TRAIN_ENV_FN, 'rb') as handle:
            extended_train_env = pickle.load(handle)
        
        with open(VAL_ENV_FN, 'rb') as handle:
            val_env = pickle.load(handle)
        
        with open(TEST_ENV_FN, 'rb') as handle:
            test_env = pickle.load(handle)
        
        with open(TEST_ENV_RAW_FN, 'rb') as handle:
            test_env_raw = pickle.load(handle)
    except FileNotFoundError:
        train_env = StockEnv(
            tickers=["AAPL", "MSI", "GOOGL", "SBUX", "GOLD", "BHP", "IBDRY"],
            start_date="2010-01-01", end_date="2019-01-01", time_period=30,
            init_investment=20000, max_action=100, use_raw_prices=False
        )
        with open(TRAIN_ENV_FN, 'wb') as handle:
            pickle.dump(train_env, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        extended_train_env = StockEnv(
            tickers=["AAPL", "MSI", "GOOGL", "SBUX", "GOLD", "BHP", "IBDRY"],
            start_date="2010-01-01", end_date="2020-01-01", time_period=30,
            init_investment=20000, max_action=100, use_raw_prices=False
        )
        with open(EXTENDED_TRAIN_ENV_FN, 'wb') as handle:
            pickle.dump(extended_train_env, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        val_env = StockEnv(
            tickers=["AAPL", "MSI", "GOOGL", "SBUX", "GOLD", "BHP", "IBDRY"],
            start_date="2019-01-01", end_date="2020-01-01", time_period=30,
            init_investment=20000, max_action=100, use_raw_prices=False
        )
        with open(VAL_ENV_FN, 'wb') as handle:
            pickle.dump(val_env, handle, protocol=pickle.HIGHEST_PROTOCOL)

        test_env = StockEnv(
            tickers=["AAPL", "MSI", "GOOGL", "SBUX", "GOLD", "BHP", "IBDRY"],
            start_date="2020-01-01", end_date="2022-08-25", time_period=30,
            init_investment=20000, max_action=100, use_raw_prices=False
        )
        with open(TEST_ENV_FN, 'wb') as handle:
            pickle.dump(test_env, handle, protocol=pickle.HIGHEST_PROTOCOL)

        test_env_raw = StockEnv(
            tickers=["AAPL", "MSI", "GOOGL", "SBUX", "GOLD", "BHP", "IBDRY"],
            start_date="2020-01-01", end_date="2022-08-25", time_period=30,
            init_investment=20000, max_action=100, use_raw_prices=True
        )
        with open(TEST_ENV_RAW_FN, 'wb') as handle:
            pickle.dump(test_env_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return train_env, extended_train_env, val_env, test_env, test_env_raw




if __name__ == "__main__":
    train_env, extended_train_env, val_env, test_env, test_env_raw = load_envs()

    reward, portfolio_values_baseline = run_baseline(test_env_raw)
    print("Baseline Agent Reward on Test env", reward)

    # TODO: FINISH FROM HERE
    train_rbf = RBFTransform()


    

        