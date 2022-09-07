import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # skopt np.int deprecation warning

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from typing import List, Union, Type, Tuple

from stock_trading.baseline_agent import BaselineAgent
from stock_trading.stock_env import StockEnv
from light_rl.common.base.agent import BaseAgent
from light_rl.common.transforms.rbf import RBFTransform
from light_rl.common.transforms.standard_scaler import StandardScalerTransform
from light_rl.common.plotting import plot_avg_reward


EPISODIC_R_PLT_FN     = "episodic_reward.pdf"
EPISODIC_R_T_PLT_FN   = "episodic_reward_wall_clock.pdf"
AVG_EPISODE_R_PLT_FN  = "episodic_reward_avg.pdf"

class CategoricalList(Categorical):
    def __init__(self, categories, **categorical_kwargs):
        super().__init__(self._convert_hashable(categories), **categorical_kwargs)       
        
    def _convert_hashable(self, list_of_lists):
        return [_HashableListAsDict(list_) 
                for list_ in list_of_lists]


class _HashableListAsDict(dict):
    def __init__(self, arr):
        self.idx = 0
        self.update({i:val for i, val in enumerate(arr)})

    def __hash__(self):
        return hash(tuple(sorted(self.items())))   
    
    def __repr__(self):
        return str(list(self.values()))
    
    def __getitem__(self, key):
        return list(self.values())[key]
    
    def __iter__(self):
        return _ListIter(self)
    
    def __eq__(self, rhs):
        if len(self) != len(rhs):
            return False
        for i in range(len(self)):
            if self[i] != rhs[i]:
                return False
        return True


class _ListIter:
    def __init__(self, indexable, idx=0):
        self.indexable = indexable
        self.idx = idx
    
    def __next__(self):
        if self.idx >= len(self.indexable):
            raise StopIteration
        item = self.indexable[self.idx]
        self.idx += 1
        return item


def play_episode(env: StockEnv, agent: BaseAgent) -> Tuple[float, np.ndarray]:
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


def train_agent(search_space: List[Union[Categorical, CategoricalList, Integer, Real]],
            x0: List[List], agent_cls: Type[BaseAgent], agent_folder_name: str, 
            train_env: StockEnv, extended_train_env: StockEnv, 
            val_env: StockEnv, test_env: StockEnv, 
            rbf: RBFTransform, extended_rbf: RBFTransform,
            standardiser: StandardScalerTransform,
            extended_standardiser: StandardScalerTransform,
            train_timelimit=60*60, n_calls=100,
            train_episodes=5000, extended_train_episodes=10000,
            eval_freq_episodes=100) -> Tuple[float, np.ndarray]:
    """ Train agent. Tune hyperparams on val env.
    Train on extended_train_env with best hyperparam setting
    and then eval on test_env. Return reward and portfolio values 
    for single episode on test_env. Will only train an agent
    if a saved agent does not exist in agent_folder_name.

    Args:
        search_space (List[Union[Categorical, CategoricalList, Integer, Real]]):
            search space used for bayesian hyperparameter tuning. Must include
            Catagorical with name "use_rbf".
        x0 (List[List[Any]]): list of initial hyperparameters to try.
        agent_cls (Type[BaseAgent]): agent class, used to instantiate agents.
        agent_folder_name (str): path to folder name, will save agent files here.
        train_env (StockEnv): training env
        extended_train_env (StockEnv): training env for best hyperparam setting
            (included validation data)
        val_env (StockEnv): validation env
        test_env (StockEnv): test env
        rbf (RBFTransform): RBF feature transformer for train env
        extended_rbf (RBFTransform): RBF for extended_train_env
        standardiser (StandardScalerTransform): standard scaler feature
            transformer for train env
        extended_standiser (StandardScalerTransform): standard scaler
            feature transformer for extended_train_env
        train_timelimit (int, optional): max amount of time in seconds
            spent training a single model. 
        n_calls (int, optional): number of models to try during
            hyperparameter optimisation.
        train_episodes (int, optional): number of episodes to train
            a single model.
        extended_train_episodes (int, optional): number of episodes
            to train the final model on extended_train_env.
        eval_freq_episodes (int, optional): evaluation frequency
            in terms of episodes.

    Returns:
        episode_reward (float): reward from the test env episode
        portfolio_val_hist (np.ndarray): np array of
            portfolio values with shape (num_trading_days, )
    """
    try:
        agent = agent_cls.load(agent_folder_name)
    except FileNotFoundError:
        # find best hyperparam on val set
        train_episode_length = train_env.stock_prices.shape[0] - 1
        train_config = {
            "env": train_env,
            "max_timesteps": train_episodes * train_episode_length,
            "max_training_time": train_timelimit,
            "target_return": float("inf"),
            "max_episode_length": float("inf"),
            "eval_freq": eval_freq_episodes * train_episode_length,
            "eval_episodes": 1,
        }
        evaluate_agent_data = {
            "counter": 0,
            "best_agent": None,
            "best_reward": float("-inf")
        }

        @use_named_args(search_space)
        def evaluate_agent(**params):
            evaluate_agent_data["counter"] += 1
            kwargs = {
                "ft_transformer": standardiser if params['use_rbf'] == 'false' else rbf,
                "action_space": train_env.action_space,
                "state_space": train_env.observation_space
            }
            params.pop('use_rbf')
            kwargs = {**kwargs, **params}
            agent = agent_cls(**kwargs)
            print(f"Training {agent_cls} agent {evaluate_agent_data['counter']}")
            agent.train(**train_config)

            # play on validation set
            reward, portfilio_values = play_episode(val_env, agent)

            if reward > evaluate_agent_data["best_reward"]:
                evaluate_agent_data["best_agent"] = agent
                evaluate_agent_data["best_reward"] = reward

            # convert from a maximizing score to a minimizing score
            return -reward
        
        res = gp_minimize(evaluate_agent, search_space, x0=x0, n_calls=n_calls)

        # train on extended training env (contains validation data)
        best_agent = evaluate_agent_data["best_agent"]
        best_kwargs = best_agent.init_kwargs
        if isinstance(best_kwargs["ft_transformer"], StandardScalerTransform):
            best_kwargs["ft_transformer"] = extended_standardiser
        else:
            best_kwargs["ft_transformer"] = extended_rbf
        
        # init agent with old agents params, train on extended env
        print("Hyperparameter tuning done")
        print("best_val_reward:", {evaluate_agent_data['best_reward']})
        print("best params:", best_kwargs)
        
        extended_episode_length = extended_train_env.stock_prices.shape[0] - 1
        agent = agent_cls(**best_kwargs)

        rewards, times = agent.train(
            env=extended_train_env,
            max_timesteps=extended_train_episodes * extended_episode_length,
            max_training_time=train_timelimit,
            target_return=float("inf"),
            max_episode_length=float("inf"),
            eval_freq=eval_freq_episodes * extended_episode_length,
            eval_episodes=1
        )
        agent.save(agent_folder_name)

        # plot episodic return against episode number
        fig = plt.figure()
        plt.plot(rewards)
        plt.title("Episodic Reward")
        plt.xlabel("Episode Number")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(agent_folder_name, EPISODIC_R_PLT_FN))
        plt.close(fig)

        # plot episodic return against wall clock time
        fig = plt.figure()
        plt.plot(times, rewards)
        plt.title("Episodic Reward")
        plt.xlabel("Elapsed Time (s)")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(agent_folder_name, EPISODIC_R_T_PLT_FN))
        plt.close(fig)

        plot_avg_reward(rewards, os.path.join(agent_folder_name, AVG_EPISODE_R_PLT_FN))
    
    # run trained agent on test env
    return play_episode(test_env, agent)