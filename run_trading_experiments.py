import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import json

from typing import Tuple

from skopt.space import Integer, Real, Categorical

from stock_trading.baseline_agent import BaselineAgent
from stock_trading.stock_env import StockEnv
from stock_trading.tests import tests
from light_rl.algorithms.a3c import A3C
from light_rl.algorithms.ddpg import DDPG
from light_rl.algorithms.es import ES
from light_rl.algorithms.vanilla_policy_gradient import VanillaPolicyGradient
from light_rl.common.transforms.rbf import RBFTransform
from light_rl.common.transforms.standard_scaler import StandardScalerTransform
from bayesian_optimization import play_episode, train_agent, CategoricalList, _HashableListAsDict


FOLDER_FN             = "stock_files"
TRAIN_ENV_FN          = os.path.join(FOLDER_FN, "train_env.pkl")
EXTENDED_TRAIN_ENV_FN = os.path.join(FOLDER_FN, "extended_train_env.pkl")
VAL_ENV_FN            = os.path.join(FOLDER_FN, "val_env.pkl")
TEST_ENV_FN           = os.path.join(FOLDER_FN, "test_env.pkl")
TEST_ENV_RAW_FN       = os.path.join(FOLDER_FN, "test_env_raw.pkl")
DJI_TEST_PERIOD_FN    = os.path.join(FOLDER_FN, "dji_test_period.pkl")
DDPG_FN               = os.path.join(FOLDER_FN, "ddpg_model")
ES_FN                 = os.path.join(FOLDER_FN, "es_model")
A3C_FN                = os.path.join(FOLDER_FN, "a3c_model")
PG_FN                 = os.path.join(FOLDER_FN, "pg_model")
PORTFOLIO_VALS_PLT_FN = os.path.join(FOLDER_FN, "portfolio_values.pdf")
TEST_REWARDS_FN       = os.path.join(FOLDER_FN, "test_rewards.txt")
# 3 hr time limit to train each model, once hyperparams have been tuned
TRAIN_TIMELIMIT       = 3*60*60


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


def run_DDPG(train_env: StockEnv, extended_train_env: StockEnv, 
             val_env: StockEnv, test_env: StockEnv, 
             rbf: RBFTransform, extended_rbf: RBFTransform,
             standardiser: StandardScalerTransform,
             extended_standardiser: StandardScalerTransform,
             train_timelimit=60*60, n_calls=100,
             train_episodes=5000, extended_train_episodes=10000,
             eval_freq_episodes=100) -> Tuple[float, np.ndarray]:
    search_space = [
        Real(0.97, 1.0, 'uniform', name='gamma'),
        Real(5e-5, 1e-3, 'uniform', name='actor_lr'),
        Real(5e-5, 1e-3, 'uniform', name='critic_lr'),
        Real(1e-8, 1e-3, 'uniform', name='actor_adam_eps'),
        Real(1e-8, 1e-3, 'uniform', name='critic_adam_eps'),
        Real(0, 1e-2, 'uniform', name='actor_weight_decay'),
        Real(0, 1e-2, 'uniform', name='critic_weight_decay'),
        Real(5e-3, 1e-1, 'uniform', name='noise_std'),
        Integer(16, 256, "uniform", name="batch_size"),
        Integer(int(2e5), int(2e6), "log-uniform", name="buffer_capacity"),
        Categorical(['true', 'false'], name="use_rbf"),
        CategoricalList([
            [64, 64], [128, 128],
            [16, 16, 16, 16], [32, 32, 32, 32], [64, 64, 64, 64],
            [128, 128, 128, 128]
            ], name="actor_hidden_layers"
        ),
        CategoricalList([
            [64, 64], [128, 128],
            [16, 16, 16, 16], [32, 32, 32, 32], [64, 64, 64, 64],
            [128, 128, 128, 128]
            ], name="critic_hidden_layers"
        )
    ]
    x0 = [
        [0.99, 1e-3, 1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-2, 64, 
        int(1e6), "false", _HashableListAsDict([64, 64]), 
        _HashableListAsDict([64, 64])],
        [0.9999, 1e-4, 1e-4, 1e-8, 1e-8, 0, 0, 1e-2, 64, 
        int(1e6), "false", _HashableListAsDict([64, 64]), 
        _HashableListAsDict([64, 64])]
    ]
    return train_agent(
        search_space=search_space, 
        x0=x0, 
        agent_cls=DDPG, 
        agent_folder_name=DDPG_FN, 
        train_env=train_env, 
        extended_train_env=extended_train_env, 
        val_env=val_env, 
        test_env=test_env, 
        rbf=rbf, 
        extended_rbf=extended_rbf, 
        standardiser=standardiser, 
        extended_standardiser=extended_standardiser, 
        train_timelimit=train_timelimit, 
        n_calls=n_calls, 
        train_episodes=train_episodes, 
        extended_train_episodes=extended_train_episodes, 
        eval_freq_episodes=eval_freq_episodes
    )


def run_ES(train_env: StockEnv, extended_train_env: StockEnv, 
           val_env: StockEnv, test_env: StockEnv, 
           rbf: RBFTransform, extended_rbf: RBFTransform,
           standardiser: StandardScalerTransform,
           extended_standardiser: StandardScalerTransform,
           train_timelimit=60*60, n_calls=100,
           train_episodes=5000, extended_train_episodes=10000,
           eval_freq_episodes=100) -> Tuple[float, np.ndarray]:
    search_space = [
        Real(1e-2, 2e-1, 'uniform', name='lr'),
        Real(1e-1, 1.0, 'uniform', name='std_noise'),
        Real(1e-8, 1e-3, 'uniform', name='actor_adam_eps'),
        Real(0, 1e-2, 'uniform', name='actor_weight_decay'),
        Integer(64, 512, "uniform", name="lstm_hidden_dim"),
        Integer(20, 250, "uniform", name="pop_size"),
        Categorical(['true', 'false'], name="use_rbf"),
        CategoricalList([
            [16, 16, 16, 16], [64, 64],
            [128, 'lstm'], [64, 64, 'lstm'], [64, 'lstm', 64], 
            [128, 'lstm', 'lstm']
            ], name="actor_hidden_layers"
        )
    ]
    x0 = [
        [0.1, 0.5, 1e-8, 1e-5, 256, 50, "false", _HashableListAsDict([16, 16, 16, 16])],
        [0.1, 0.5, 1e-8, 1e-5, 256, 50, "false", _HashableListAsDict([64, 64])],
        [0.1, 0.5, 1e-8, 1e-5, 256, 50, "false", _HashableListAsDict([64, 64, 'lstm'])],
        [0.1, 0.5, 1e-8, 1e-5, 256, 50, "false", _HashableListAsDict([128, 'lstm', 'lstm'])]
    ]
    return train_agent(
        search_space=search_space, 
        x0=x0, 
        agent_cls=ES, 
        agent_folder_name=ES_FN, 
        train_env=train_env, 
        extended_train_env=extended_train_env, 
        val_env=val_env, 
        test_env=test_env, 
        rbf=rbf, 
        extended_rbf=extended_rbf, 
        standardiser=standardiser, 
        extended_standardiser=extended_standardiser, 
        train_timelimit=train_timelimit, 
        n_calls=n_calls, 
        train_episodes=train_episodes, 
        extended_train_episodes=extended_train_episodes, 
        eval_freq_episodes=eval_freq_episodes
    )


def run_A3C(train_env: StockEnv, extended_train_env: StockEnv, 
            val_env: StockEnv, test_env: StockEnv, 
            rbf: RBFTransform, extended_rbf: RBFTransform,
            standardiser: StandardScalerTransform,
            extended_standardiser: StandardScalerTransform,
            train_timelimit=60*60, n_calls=100,
            train_episodes=5000, extended_train_episodes=10000,
            eval_freq_episodes=100) -> Tuple[float, np.ndarray]:
    search_space = [
        Real(0.99, 1.0, 'uniform', name='gamma'),
        Real(5e-5, 1e-3, 'uniform', name='actor_lr'),
        Real(5e-5, 1e-3, 'uniform', name='critic_lr'),
        Real(1e-8, 1e-3, 'uniform', name='actor_adam_eps'),
        Real(1e-8, 1e-3, 'uniform', name='critic_adam_eps'),
        Real(0, 1e-2, 'uniform', name='actor_weight_decay'),
        Real(0, 1e-2, 'uniform', name='critic_weight_decay'),
        Real(0, 1e-3, 'uniform', name='entropy_beta'),
        Integer(1, 50, "uniform", name="tmax"),
        Categorical(['true', 'false'], name="use_rbf"),
        CategoricalList([
            [], [64, 64], [128, 128],
            [16, 16, 16, 16], [32, 32, 32, 32], [64, 64, 64, 64],
            [128, 128, 128, 128], [64, 'lstm']
            ], name="actor_hidden_layers"
        ),
        CategoricalList([
            [], [64, 64], [128, 128],
            [16, 16, 16, 16], [32, 32, 32, 32], [64, 64, 64, 64],
            [128, 128, 128, 128], [64, 'lstm']
            ], name="critic_hidden_layers"
        )
    ]
    x0 = [
        [0.9999, 1e-4, 1e-4, 1e-3, 1e-3, 1e-6, 1e-6, 1e-5, 20, "false", 
        _HashableListAsDict([64, 64]), _HashableListAsDict([64, 64])],
        [0.9999, 1e-4, 1e-4, 1e-3, 1e-3, 1e-6, 1e-6, 1e-5, 20, "true", 
        _HashableListAsDict([64, 64]), _HashableListAsDict([64, 64])],
        [0.9999, 1e-4, 1e-4, 1e-3, 1e-3, 1e-6, 1e-6, 1e-5, 20, "true", 
        _HashableListAsDict([64, 64]), _HashableListAsDict([64, 'lstm'])],
        [0.9999, 1e-4, 1e-4, 1e-3, 1e-3, 1e-6, 1e-6, 1e-5, 20, "true", 
        _HashableListAsDict([]), _HashableListAsDict([])]
    ]
    return train_agent(
        search_space=search_space, 
        x0=x0, 
        agent_cls=A3C, 
        agent_folder_name=A3C_FN, 
        train_env=train_env, 
        extended_train_env=extended_train_env, 
        val_env=val_env, 
        test_env=test_env, 
        rbf=rbf, 
        extended_rbf=extended_rbf, 
        standardiser=standardiser, 
        extended_standardiser=extended_standardiser, 
        train_timelimit=train_timelimit, 
        n_calls=n_calls, 
        train_episodes=train_episodes, 
        extended_train_episodes=extended_train_episodes, 
        eval_freq_episodes=eval_freq_episodes
    )


def run_PG(train_env: StockEnv, extended_train_env: StockEnv, 
           val_env: StockEnv, test_env: StockEnv, 
           rbf: RBFTransform, extended_rbf: RBFTransform,
           standardiser: StandardScalerTransform,
           extended_standardiser: StandardScalerTransform,
           train_timelimit=60*60, n_calls=100,
           train_episodes=5000, extended_train_episodes=10000,
           eval_freq_episodes=100) -> Tuple[float, np.ndarray]:
    search_space = [
        Real(0.99, 1.0, 'uniform', name='gamma'),
        Real(5e-5, 1e-3, 'uniform', name='actor_lr'),
        Real(5e-5, 1e-3, 'uniform', name='critic_lr'),
        Real(1e-8, 1e-3, 'uniform', name='actor_adam_eps'),
        Real(1e-8, 1e-3, 'uniform', name='critic_adam_eps'),
        Real(0, 1e-2, 'uniform', name='actor_weight_decay'),
        Real(0, 1e-2, 'uniform', name='critic_weight_decay'),
        Categorical(['true', 'false'], name="use_rbf"),
        CategoricalList([
            [], ['lstm'], [64, 64], [64, 'lstm']
            ], name="actor_hidden_layers"
        ),
        CategoricalList([
            [], ['lstm'], [64, 64], [64, 'lstm']
            ], name="critic_hidden_layers"
        )
    ]
    x0 = [
        [0.9999, 1e-4, 1e-4, 1e-3, 1e-3, 1e-6, 1e-6, "true", 
        _HashableListAsDict([64, 64]), _HashableListAsDict([64, 64])],
        [0.9999, 1e-4, 1e-4, 1e-3, 1e-3, 1e-6, 1e-6, "true", 
        _HashableListAsDict(['lstm']), _HashableListAsDict(['lstm'])],
        [0.9999, 1e-4, 1e-4, 1e-3, 1e-3, 1e-6, 1e-6, "true", 
        _HashableListAsDict([]), _HashableListAsDict([])]
    ]
    return train_agent(
        search_space=search_space, 
        x0=x0,
        agent_cls=VanillaPolicyGradient, 
        agent_folder_name=PG_FN, 
        train_env=train_env, 
        extended_train_env=extended_train_env, 
        val_env=val_env, 
        test_env=test_env, 
        rbf=rbf, 
        extended_rbf=extended_rbf, 
        standardiser=standardiser, 
        extended_standardiser=extended_standardiser, 
        train_timelimit=train_timelimit, 
        n_calls=n_calls, 
        train_episodes=train_episodes, 
        extended_train_episodes=extended_train_episodes, 
        eval_freq_episodes=eval_freq_episodes
    )


def load_envs():
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
            init_investment=10000, max_action=100, use_raw_prices=False
        )
        with open(TRAIN_ENV_FN, 'wb') as handle:
            pickle.dump(train_env, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        extended_train_env = StockEnv(
            tickers=["AAPL", "MSI", "GOOGL", "SBUX", "GOLD", "BHP", "IBDRY"],
            start_date="2010-01-01", end_date="2020-01-01", time_period=30,
            init_investment=10000, max_action=100, use_raw_prices=False
        )
        with open(EXTENDED_TRAIN_ENV_FN, 'wb') as handle:
            pickle.dump(extended_train_env, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        val_env = StockEnv(
            tickers=["AAPL", "MSI", "GOOGL", "SBUX", "GOLD", "BHP", "IBDRY"],
            start_date="2019-01-01", end_date="2020-01-01", time_period=30,
            init_investment=10000, max_action=100, use_raw_prices=False
        )
        with open(VAL_ENV_FN, 'wb') as handle:
            pickle.dump(val_env, handle, protocol=pickle.HIGHEST_PROTOCOL)

        test_env = StockEnv(
            tickers=["AAPL", "MSI", "GOOGL", "SBUX", "GOLD", "BHP", "IBDRY"],
            start_date="2020-01-01", end_date="2022-08-25", time_period=30,
            init_investment=10000, max_action=100, use_raw_prices=False
        )
        with open(TEST_ENV_FN, 'wb') as handle:
            pickle.dump(test_env, handle, protocol=pickle.HIGHEST_PROTOCOL)

        test_env_raw = StockEnv(
            tickers=["AAPL", "MSI", "GOOGL", "SBUX", "GOLD", "BHP", "IBDRY"],
            start_date="2020-01-01", end_date="2022-08-25", time_period=30,
            init_investment=10000, max_action=100, use_raw_prices=True
        )
        with open(TEST_ENV_RAW_FN, 'wb') as handle:
            pickle.dump(test_env_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return train_env, extended_train_env, val_env, test_env, test_env_raw


def calc_anualized_return(first_value: float, final_value: float, days_held: int):
    cumulative_ret = (final_value - first_value) / first_value
    return ((1 + cumulative_ret) ** (365 / days_held)) - 1


def calc_sharpe_ratio(portfolio_values: np.ndarray):
    r = np.diff(portfolio_values)
    return r.mean() / r.std() * np.sqrt(252)


if __name__ == "__main__":
    train_env, extended_train_env, val_env, test_env, test_env_raw = load_envs()
    baseline_reward, baseline_portfolio_values = run_baseline(test_env_raw)

    # create Transforms
    transforms_n_episodes = 750
    transforms_n_components = 500
    rbf = RBFTransform(
        train_env, n_episodes=transforms_n_episodes, 
        n_components=transforms_n_components
    )
    extended_rbf = RBFTransform(
        extended_train_env, n_episodes=transforms_n_episodes, 
        n_components=transforms_n_components
    )
    standardiser = StandardScalerTransform(
        train_env, n_episodes=transforms_n_episodes
    )
    extended_standiser = StandardScalerTransform(
        extended_train_env, n_episodes=transforms_n_episodes
    )

    # train agents (this may take some time), 
    # increase n_calls to try more hyperparameters 
    # (18 probs isnt enough but dont have strong hardware :( )
    ddpg_reward, ddpg_portfolio_values = run_DDPG(
        train_env, extended_train_env, val_env, test_env, 
        rbf, extended_rbf, standardiser, extended_standiser,
        train_timelimit=TRAIN_TIMELIMIT, n_calls=18,
        train_episodes=150, extended_train_episodes=1000,
        eval_freq_episodes=50
    )

    pg_reward, pg_portfolio_values = run_PG(
        train_env, extended_train_env, val_env, test_env, 
        rbf, extended_rbf, standardiser, extended_standiser,
        train_timelimit=TRAIN_TIMELIMIT, n_calls=18,
        train_episodes=150, extended_train_episodes=1000,
        eval_freq_episodes=50
    )

    es_reward, es_portfolio_values = run_ES(
        train_env, extended_train_env, val_env, test_env, 
        rbf, extended_rbf, standardiser, extended_standiser,
        train_timelimit=TRAIN_TIMELIMIT, n_calls=18,
        train_episodes=5e5, extended_train_episodes=1e8,
        eval_freq_episodes=1000
    )

    a3c_reward, a3c_portfolio_values = run_A3C(
        train_env, extended_train_env, val_env, test_env, 
        rbf, extended_rbf, standardiser, extended_standiser,
        train_timelimit=TRAIN_TIMELIMIT, n_calls=18,
        train_episodes=1000, extended_train_episodes=1e6,
        eval_freq_episodes=500
    )

    # plot test env portfolio values for baseline, DDPG, ES, A3C, PG, DJI on same graph
    x = np.arange(baseline_portfolio_values.shape[0])
    fig = plt.figure()
    plt.plot(x, baseline_portfolio_values, 'b', label="Baseline") 
    plt.plot(x, ddpg_portfolio_values, 'g', label="DDPG")
    plt.plot(x, pg_portfolio_values, 'r', label="VPG")
    plt.plot(x, es_portfolio_values, 'm', label="ES")
    plt.plot(x, a3c_portfolio_values, 'y', label="A3C")
    plt.title("Portfolio Values")
    plt.xlabel("Trading Day")
    plt.ylabel("$ Value")
    plt.legend(loc="best")
    plt.savefig(PORTFOLIO_VALS_PLT_FN)
    plt.close(fig)

    # days held is the number of days between start date and end date (note this is not trading days)
    days_held = (test_env.dates[-1] - test_env.dates[0]).astype('timedelta64[D]')  / np.timedelta64(1, 'D')
    first_val = baseline_portfolio_values[0]
    json_data = {
        "baseline_reward": baseline_reward,
        "baseline_final_value": baseline_portfolio_values[-1],
        "baseline_anualized_return": calc_anualized_return(
            first_val, baseline_portfolio_values[-1], days_held
        ),
        "baseline_sharpe_ratio": calc_sharpe_ratio(baseline_portfolio_values),
        "ddpg_reward": ddpg_reward,
        "ddpg_final_value": ddpg_portfolio_values[-1],
        "ddpg_anualized_return": calc_anualized_return(
            first_val, ddpg_portfolio_values[-1], days_held
        ),
        "ddpg_sharpe_ratio": calc_sharpe_ratio(ddpg_portfolio_values),
        "pg_reward": pg_reward,
        "pg_final_value": pg_portfolio_values[-1],
        "pg_anualized_return": calc_anualized_return(
            first_val, pg_portfolio_values[-1], days_held
        ),
        "pg_sharpe_ratio": calc_sharpe_ratio(pg_portfolio_values),
        "es_reward": es_reward,
        "es_final_value": es_portfolio_values[-1],
        "es_anualized_return": calc_anualized_return(
            first_val, es_portfolio_values[-1], days_held
        ),
        "es_sharpe_ratio": calc_sharpe_ratio(es_portfolio_values),
        "a3c_reward": a3c_reward,
        "a3c_final_value": a3c_portfolio_values[-1],
        "a3c_anualized_return": calc_anualized_return(
            first_val, a3c_portfolio_values[-1], days_held
        ),
        "a3c_sharpe_ratio": calc_sharpe_ratio(a3c_portfolio_values),
    }
    with open(TEST_REWARDS_FN, 'w') as out_file:
        json.dump(json_data, out_file, sort_keys=True, indent=4,
               ensure_ascii=False)


        