import time
import gym
import torch
import numpy as np

from torch.optim import Adam
from torch.distributions.normal import Normal
from tqdm import tqdm

from light_rl.common.base.agent import BaseAgent
from light_rl.common.nets.device import DEVICE
from light_rl.common.nets.fc_network import FCNetwork
from light_rl.common.base.feature_transform import Transform
from light_rl.common.replay_buffer import ReplayBuffer, Transition


class DDPG(BaseAgent):
    def __init__(self, action_space: gym.Space, state_space: gym.Space,
            ft_transformer = Transform(), 
            actor_hidden_layers=[64, 64], critic_hidden_layers=[64, 64], 
            actor_lr=1e-3, critic_lr=1e-3, noise_std=0.1, gamma=0.999, tau=0.01, 
            max_grad_norm=50, batch_size=64, buffer_capacity=int(1e6)):
        """ Constructor for DDPG agent. Assumes continious action spaces.

        Args:
            action_space (gym.Space): action space for this agent
            state_space (gym.Space): state space for this agent
            ft_transformer (Transform, optional): feature transformer. Defaults to Transform() 
            actor_hidden_layers (list, optional): actor hidden layers. Defaults to [64, 64].
            critic_hidden_layers (list, optional): critic hidden layers. Defaults to [64, 64].
            actor_lr (_type_, optional): actor learning rate. Defaults to 1e-3.
            critic_lr (_type_, optional): critic learning rate. Defaults to 1e-3.
            noise_std (float, optional): standard deviation of gaussian noise
                used for exploration during training. Defaults to 0.1.
            gamma (float, optional): discount factor gamma. Defaults to 0.999.
            tau (float, optional): tau used for target net soft updates. Defaults to 0.01.
            max_grad_norm (int, optional): max grad norm used in gradient clipping. Defaults to 50.
            batch_size (int, optional): batch size. Defaults to 64.
            buffer_capacity (_type_, optional): num sumples that replay buffer can store. Defaults to int(1e6).
        """
        super().__init__(action_space, state_space)
        self.init_kwargs = {k: v for k, v in locals().items() if k != "self"}
        self.ft_transformer = ft_transformer
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # init models
        ACTION_SIZE = action_space.shape[0]
        STATE_SIZE = state_space.shape[0]

        self.actor_net = FCNetwork(
            (STATE_SIZE, *actor_hidden_layers, ACTION_SIZE), torch.nn.Tanh 
        )
        self.actor_target_net = FCNetwork(
            (STATE_SIZE, *actor_hidden_layers, ACTION_SIZE), torch.nn.Tanh 
        )
        self.actor_target_net.hard_update(self.actor_net)
        self.actor_optim = Adam(
            self.actor_net.parameters(), lr=actor_lr, eps=1e-3
        )

        self.critic_net = FCNetwork(
            (ACTION_SIZE + STATE_SIZE, *critic_hidden_layers, 1)
        )
        self.critic_target_net = FCNetwork(
            (ACTION_SIZE + STATE_SIZE, *critic_hidden_layers, 1)
        )
        self.critic_target_net.hard_update(self.critic_net)
        self.critic_optim = Adam(
            self.critic_net.parameters(), lr=critic_lr, eps=1e-3
        )

        self.noise = Normal(torch.tensor([0.0]), torch.tensor([noise_std]))
    
    def _transform_state(self, s: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            self.ft_transformer.transform(s)
        ).to(DEVICE).float()
    
    def get_action(self, s: np.ndarray, explore: bool = False, rec_state=None):
        with torch.no_grad():
            s = self._transform_state(s)
            a = self.actor_net(s)
            if explore:
                a += torch.reshape(self.noise.sample(a.shape), a.shape)
        return a.numpy(), rec_state

    def single_online_learn_step(self, batch: Transition):
        q_loss = torch.tensor([0.0])
        p_loss = torch.tensor([0.0])
        batch_size = batch.states.shape[0]

        batch = Transition(
            self._transform_state(batch.states), 
            torch.from_numpy(batch.actions).to(DEVICE), 
            self._transform_state(batch.next_states), 
            torch.from_numpy(batch.rewards).to(DEVICE), 
            torch.from_numpy(batch.terminal).to(DEVICE)
        )

        # first update critic by minimising MSE of TD error on batch
        with torch.no_grad():
            q_values =  self.critic_target_net(
                torch.concat(
                    (batch.next_states, self.actor_target_net(batch.next_states)), 
                    axis=-1
                )
            )
            targets = batch.rewards + (self.gamma * (1 - batch.terminal) * q_values) 
        
        y_pred = self.critic_net(torch.concat((batch.states, batch.actions), axis=-1))
        q_loss = (targets - y_pred).pow(2).sum() / batch_size

        self.critic_optim.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        # now update actor by maxmizing q value from critic
        q_values = self.critic_net(
            torch.concat((batch.states, self.actor_net(batch.states)), axis=-1)
        )
        p_loss = (-q_values).sum() / batch_size

        self.actor_optim.zero_grad()
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        # update the target networks
        self.critic_target_net.soft_update(self.critic_net, self.tau)
        self.actor_target_net.soft_update(self.actor_net, self.tau)
    
    def train(self, env: gym.Env, max_timesteps: int, max_training_time: float, 
              target_return: float, max_episode_length: int, eval_freq: int, eval_episodes: int) -> list:
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

                    self.replay_buffer.push(
                        np.array(s, dtype=np.float32),
                        np.array(a, dtype=np.float32),
                        np.array(next_s, dtype=np.float32),
                        np.array([r], dtype=np.float32),
                        np.array([terminal], dtype=np.float32),
                    )
                    if len(self.replay_buffer) >= self.batch_size:
                        batch = self.replay_buffer.sample(self.batch_size)
                        self.single_online_learn_step(batch)
                    
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