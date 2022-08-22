import os
import numpy as np
import gym
import torch
import time
import pickle

from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Any
from tqdm import tqdm



class BaseAgent(ABC):
    TORCH_SAVEABLES_FN = "torch_savables.pt"
    OTHER_SAVEABLES_FN = "other_savebles.pkl"
    KWARGS_FN = "kwargs.pkl"

    def __init__(self, action_space: gym.Space, state_space: gym.Space):
        """Constructor 

        Args:
            action_space (gym.Space): action space of the enviroment
            state_space (gym.space): state space of the enviroment
        """
        self.init_kwargs = {k: v for k, v in locals().items() if k != "self"}  # would need to put this subclass

        self.action_space = action_space
        self.state_space = state_space

        self.torch_saveables = {}
        self.other_saveables = {}  # optional, for any additional hidden state
        
    @abstractmethod
    def get_action(self, state: np.ndarray, explore: bool, rec_state: Any = None) -> Tuple[np.ndarray, Any]:
        """ Get an action given state using agent policy

        Args:
            state (np.ndarray): state returned by enviroment
            explore (bool): whether or not agent should explore 
                (e.g. should use epsilon greedy policy).
                This should be true during training
            rec_state (Any): recurrent state, used by actor-critics
                that utilise lstm. For initial state can pass None.
        Returns:
            Tuple[np.ndarray, Any]: first item is action, second is recurrent state,
                pass this to next call of get_action
        """
        pass
    
    def train(self, env: gym.Env, max_timesteps: int, max_training_time: float, 
              target_return: float, max_episode_length: int, eval_freq: int, eval_episodes: int) -> Tuple[List[float], List[float]]:
        """ Train the agent on a given enviroment.

        Args:
            env (gym.Env): Enviroment used to train the agent. Can be any enviroment
                as long as it follows the new_step_api gym interface. IMPORTANT
                MUST USE THE NEW_STEP_API.
            max_timesteps (int): maximum timesteps before training finishes
            max_training_time (float): maximum training time before training finishes
            target_return (float): if this return is exceeded when evaulating agent
                which occurs every eval_freq timesteps, then stop training
            max_episode_length (int): maximum length of episode
            eval_freq (int): evaluation of agent occurs every eval_freq timesteps,
                during evaluation any exploration would be turned off (e.g. epsilon = 0)
            eval_episodes (int): number of episodes used for evaluation. The average
                return from these episodes is taken and checked to see if this
                exceeds target return. If so end the episode.
        Returns:
            (Tuple[List[float], List[float]]): first is list of episodic return ,
                second is wall clock time since start for each respective episodic return
        """
        # default implementation. Assumes agent has a functional single_online_learn_step method.
        episode_rewards = []
        episode_r_times = []
        start_time = time.time()
        timesteps_elapsed = 0

        with tqdm(total=max_timesteps) as progress_bar:
            while timesteps_elapsed < max_timesteps:

                # play episode
                done = False
                self.reset()
                s = env.reset()
                episode_reward = 0
                iters = 0
                while not done and iters < max_episode_length:
                    a, _ = self.get_action(s, True)
                    next_s, r, terminal, truncated, info = env.step(a)
                    done = terminal
                    
                    self.single_online_learn_step(s, a, r, next_s, terminal)

                    s = next_s
                    episode_reward += r
                    iters += 1
                
                elapsed_time = time.time() - start_time
                episode_rewards.append(episode_reward)
                episode_r_times.append(elapsed_time)
                timesteps_elapsed += iters
                progress_bar.update(iters)

                if timesteps_elapsed % eval_freq < iters:
                    avg_r = 0
                    for _ in range(eval_episodes):
                        avg_r += self.play_episode(env, max_episode_length)[0]
                    avg_r /= eval_episodes

                    progress_bar.write(
                        f"Step {timesteps_elapsed} at time {round(elapsed_time, 3)}s. "
                        f"Average reward using {eval_episodes} evals: {avg_r}"
                    )

                    if avg_r >= target_return:
                        tqdm.write(f"Avg reward {avg_r} >= Target reward {target_return}, stopping early.")
                        break
                
                if elapsed_time > max_training_time:
                    progress_bar.write(f"Training time limit reached. Training stopped at {elapsed_time}s.")
                    break
        
        return episode_rewards, episode_r_times
    

    def single_online_learn_step(self, s: np.ndarray, a: np.ndarray, r: float, next_s: np.ndarray, terminal: bool):
        """ Perform a single online learning step. This could be used as follows:
            while not terminal:
                a = agent.get_action(s, True)
                next_s, r, terminal, truncated, info = env.step(a)

                agent.single_online_learn_step(s, a, r, next_s, terminal)  # online learning step

                next_s = s
            
            some algorithms cannot perform a single online learning step (e.g. A3C)
            in which case this method should do nothing.
        
            Args:
                s (np.ndarray): the second to last observed state
                a (np.ndarray): the action taken at state s
                r (float): reward for taking action a at state s
                next_s (np.ndarray): the last observed state
                terminal (bool): True if next_s is a terminal state
        """
        pass

    def reset(self):
        """ Call before starting episode, so that agent can reset variables if needed
        """
        pass

    def play_episode(self, env: gym.Env, max_episode_length: int) -> Tuple[float, int]:
        """ Play a single episode on a given env using this agent

        Args:
            env (gym.Env): Enviroment to play episode on.
            max_episode_length (int): Maximum episode length.

        Returns:
            Tuple[float, int]: reward obtained on this episode, episode timesteps
        """
        done = False
        self.reset()
        s = env.reset()
        episode_reward = 0
        iters = 0
        rec_state = None

        while not done and iters < max_episode_length:
            a, rec_state = self.get_action(s, False, rec_state)
            next_s, r, terminal, truncated, info = env.step(a)
            done = terminal

            s = next_s
            episode_reward += r
            iters += 1
        
        return episode_reward, iters

    def save(self, path: str):
        """ Save model to path for later loading. Will save files:
                1. path/torch_savables.pt
                2. path/other_savables.pkl
                3. path/kwargs.pkl
        
        Args:
            path (str): path to save agent
        """
        torch.save(self.torch_saveables, os.path.sep(path, self.TORCH_SAVEABLES_FN))

        with open(os.path.sep(path, self.OTHER_SAVEABLES_FN), 'wb') as outp:
            pickle.dump(self.other_saveables, outp, pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.sep(path, self.KWARGS_FN), 'wb') as outp:
            pickle.dump(self.init_kwargs, outp, pickle.HIGHEST_PROTOCOL)

        return path
    
    def restore(self, path: str):
        """ Restore agent model in place given path.

        Args:
            path (str): path to save agent
        """
        checkpoint = torch.load(os.path.sep(path, self.TORCH_SAVEABLES_FN))
        for k, v in self.torch_saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

        with open(os.path.sep(path, self.OTHER_SAVEABLES_FN), 'rb') as inp:
            other_saveables = pickle.load(inp)
            for k, v in other_saveables.items():
                self.k = v
        
    @classmethod
    def load(cls, path: str):
        """ Load saved agent given path

        Args:
            path (str): path to save agent
        """
        with open(os.path.sep(path, cls.KWARGS_FN), 'rb') as inp:
            init_kwargs = pickle.load(inp)
        
        model = cls(**init_kwargs)
        model.restore(path)

        return model