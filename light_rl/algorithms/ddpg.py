from multiprocessing.sharedctypes import Value
import gym
import torch
import numpy as np

from torch.optim import Adam
from torch.distributions.normal import Normal

from light_rl.common.base.agent import BaseAgent
from light_rl.common.nets.device import DEVICE
from light_rl.common.nets.fc_network import  LSTM_STR
from light_rl.common.nets.fc_network_not_rec import FCNetworkNotRec
from light_rl.common.base.feature_transform import Transform
from light_rl.common.replay_buffer import ReplayBuffer, Transition


class DDPG(BaseAgent):
    def __init__(self, action_space: gym.Space, state_space: gym.Space,
            ft_transformer = Transform(), 
            actor_hidden_layers=[64, 64], critic_hidden_layers=[64, 64], 
            actor_lr=1e-3, critic_lr=1e-3, actor_adam_eps=1e-3, critic_adam_eps=1e-3,
            noise_std=0.1, gamma=0.999, tau=0.01, 
            max_grad_norm=50, batch_size=64, buffer_capacity=int(1e6)):
        """ Constructor for DDPG agent. Assumes continious action spaces.
        Supports GPU training.

        Args:
            action_space (gym.Space): action space for this agent
            state_space (gym.Space): state space for this agent
            ft_transformer (Transform, optional): feature transformer. Defaults to Transform() 
            actor_hidden_layers (list, optional): actor hidden layers. Defaults to [64, 64].
            critic_hidden_layers (list, optional): critic hidden layers. Defaults to [64, 64].
            actor_lr (float, optional): actor learning rate. Defaults to 1e-3.
            critic_lr (float, optional): critic learning rate. Defaults to 1e-3.
            actor_adam_eps (float, optional): critic learning rate. Defaults to 1e-3.
            critic_adam_eps (float, optional): critic learning rate. Defaults to 1e-3.
            noise_std (float, optional): standard deviation of gaussian noise
                used for exploration during training. Defaults to 0.1.
            gamma (float, optional): discount factor gamma. Defaults to 0.999.
            tau (float, optional): tau used for target net soft updates. Defaults to 0.01.
            max_grad_norm (int, optional): max grad norm used in gradient clipping. Defaults to 50.
            batch_size (int, optional): batch size. Defaults to 64.
            buffer_capacity (int, optional): num sumples that replay buffer can store. Defaults to int(1e6).
        """
        super().__init__(action_space, state_space)
        self.init_kwargs = {k: v for k, v in locals().items() if k != "self"}
        
        if LSTM_STR in actor_hidden_layers or LSTM_STR in critic_hidden_layers:
            raise ValueError(f"DDPG does not support '{LSTM_STR}'")

        self.ft_transformer = ft_transformer
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # init models
        ACTION_SIZE = action_space.shape[0]
        STATE_SIZE = ft_transformer.transform(np.zeros(state_space.shape)).shape[0]

        self.actor_net = FCNetworkNotRec(
            (STATE_SIZE, *actor_hidden_layers, ACTION_SIZE), torch.nn.Tanh 
        ).to(DEVICE)
        self.actor_target_net = FCNetworkNotRec(
            (STATE_SIZE, *actor_hidden_layers, ACTION_SIZE), torch.nn.Tanh 
        ).to(DEVICE)
        self.actor_target_net.hard_update(self.actor_net)
        self.actor_optim = Adam(
            self.actor_net.parameters(), lr=actor_lr, eps=actor_adam_eps
        )

        self.critic_net = FCNetworkNotRec(
            (ACTION_SIZE + STATE_SIZE, *critic_hidden_layers, 1)
        ).to(DEVICE)
        self.critic_target_net = FCNetworkNotRec(
            (ACTION_SIZE + STATE_SIZE, *critic_hidden_layers, 1)
        ).to(DEVICE)
        self.critic_target_net.hard_update(self.critic_net)
        self.critic_optim = Adam(
            self.critic_net.parameters(), lr=critic_lr, eps=critic_adam_eps
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

    def single_online_learn_step(self, s: np.ndarray, a: np.ndarray, r: float, next_s: np.ndarray, terminal: bool):
        q_loss = torch.tensor([0.0])
        p_loss = torch.tensor([0.0])

        # fill the replay buffer
        self.replay_buffer.push(
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.float32),
            np.array(next_s, dtype=np.float32),
            np.array([r], dtype=np.float32),
            np.array([terminal], dtype=np.float32),
        )
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)

        # convert np arrays in replay buffer to torch tensors
        batch = Transition(
            self._transform_state(batch.states), 
            torch.from_numpy(batch.actions).to(DEVICE), 
            self._transform_state(batch.next_states), 
            torch.from_numpy(batch.rewards).to(DEVICE), 
            torch.from_numpy(batch.terminal).to(DEVICE)
        )

        # first update critic by minimising MSE of TD error on batch
        with torch.no_grad():
            # use target nets to get targets
            q_values = self.critic_target_net(
                torch.concat(
                    (batch.next_states, self.actor_target_net(batch.next_states)), 
                    axis=-1
                )
            )
            targets = batch.rewards + (self.gamma * (1 - batch.terminal) * q_values) 
        
        y_pred = self.critic_net(torch.concat((batch.states, batch.actions), axis=-1))
        q_loss = (targets - y_pred).pow(2).sum() / self.batch_size

        self.critic_optim.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        # now update actor by maxmizing q value from critic
        q_values = self.critic_net(
            torch.concat((batch.states, self.actor_net(batch.states)), axis=-1)
        )
        p_loss = (-q_values).sum() / self.batch_size

        self.actor_optim.zero_grad()
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        # update the target networks
        self.critic_target_net.soft_update(self.critic_net, self.tau)
        self.actor_target_net.soft_update(self.actor_net, self.tau)
