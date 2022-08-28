import gym
import torch
import numpy as np

from torch.optim import Adam
from torch.distributions.normal import Normal

from light_rl.common.base.agent import BaseAgent
from light_rl.common.nets.device import DEVICE
from light_rl.common.nets.fc_network import FCNetwork, LSTM_STR
from light_rl.common.base.feature_transform import Transform
from light_rl.common.transforms.rbf import RBFTransform


class VanillaPolicyGradient(BaseAgent):
    def __init__(self,  action_space: gym.Space, state_space: gym.Space,
            ft_transformer=Transform(), 
            actor_hidden_layers=[], critic_hidden_layers=[], 
            actor_lr=1e-4, critic_lr=1e-4, actor_adam_eps=1e-3, 
            critic_adam_eps=1e-3, actor_weight_decay=0, critic_weight_decay=0,
            gamma=0.999, max_grad_norm=50, lstm_hidden_dim=256):
        """ Constructor for Vanilla Policy Gradient agent.
        Action space only continious for now (TODO: allow discrete action spaces)
        Uses spherical covariance for normal distribution with continious actions.
        Vanilla PG can do well with no hidden layers on actor and critic while using
        RBF ft_transformer. However, PG may work well if actor uses LSTM and
        critic has no hidden layers. Supports GPU training.

        Args:
            action_space (gym.Space): action space for this agent
            state_space (gym.Space): state space for this agent
            ft_transformer (Transform): feature transformer. Recommended to use
                RBF feature transformer.
            actor_hidden_layers (list, optional): actor hidden layers. Defaults to [].
            critic_hidden_layers (list, optional): critic hidden layers. Defaults to [].
                From experience LSTM does not work well with critic for vanilla PG.
            actor_lr (float, optional): actor learning rate. Defaults to 1e-4.
            critic_lr (float, optional): critic learning rate. Defaults to 1e-4.
            actor_adap_eps (float, optional): eps parameter used in actor adam optim
            critic_adam_eps (float, option): eps parameter used in critic adam optim
            actor_weight_decay (float, optional): L2 weight decay used for actor training
            critic_weight_decay (float, optional): L2 weight decay used for critic training
            gamma (float, optional): discount factor gamma. Defaults to 0.999.
            max_grad_norm (int, optional): max grad norm used in gradient clipping. Defaults to 50.
            lstm_hidden_dim (int, optional): (hx, cx) vector sizes of lstm if one is used
        """
        super().__init__(action_space, state_space)
        self.init_kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}

        self.ft_transformer = ft_transformer
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        # init models
        self.action_size = action_space.shape[0]
        self.state_size = ft_transformer.transform(np.zeros(state_space.shape)).shape[0]

        self.actor_net = FCNetwork(
            # +1 in output for the standard deviation for At ~ N(means, std * Identity)
            (self.state_size, *actor_hidden_layers, self.action_size + 1), 
            lstm_hidden_size=lstm_hidden_dim
        ).to(DEVICE)
        self.actor_optim = Adam(
            self.actor_net.parameters(), lr=actor_lr, 
            eps=actor_adam_eps, weight_decay=actor_weight_decay
        )
        self.soft_plus = torch.nn.Softplus()

        self.critic_net = FCNetwork(
            (self.state_size, *critic_hidden_layers, 1),
            lstm_hidden_size=lstm_hidden_dim
        ).to(DEVICE)
        self.critic_optim = Adam(
            self.critic_net.parameters(), lr=critic_lr, 
            eps=critic_adam_eps, weight_decay=critic_weight_decay
        )
        self.reset()

        self.torch_saveables.update(
            {
                "actor_net": self.actor_net,
                "critic_net": self.critic_net,
                "actor_optim": self.actor_optim,
                "critic_optim": self.critic_optim
            }
        )
    
    def reset(self):
        self.gamma_t = 1
        self.actor_rec_state = self.critic_rec_state = None
    
    def _transform_state(self, s: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            self.ft_transformer.transform(s)
        ).to(DEVICE).float()
    
    def get_action(self, s: np.ndarray, explore: bool = False, rec_state=None) -> np.ndarray:
        self.s = s
        s = self._transform_state(s)
        
        actor_out, self.actor_rec_state = self.actor_net((s, rec_state))
        mu = actor_out[:self.action_size]
        std = self.soft_plus(actor_out[self.action_size:])
        std = torch.nan_to_num(std, nan=100, posinf=100, neginf=1e-8)
        self.pdf = Normal(mu, std)
        self.a = self.pdf.sample()

        return self.a.numpy(), self.actor_rec_state
    
    def single_online_learn_step(self, s: np.ndarray, a: np.ndarray, r: float, next_s: np.ndarray, terminal: bool):
        # (s, a) needs to be from last call to get_action
        if not (a == self.a.numpy()).all():
            raise ValueError(f"Argument 'a' must have been returned from last call to get_action()")
        if not (s == self.s).all():
            raise ValueError(f"Argument 's' must have been argument to last call to get_action()")

        s, next_s = self._transform_state(s), self._transform_state(next_s)
        v_s, self.critic_rec_state = self.critic_net((s, self.critic_rec_state))
        
        # calculate r + gamma * v(next_s)
        with torch.no_grad():
            g = r
            if not terminal:
                v_next_s, _ = self.critic_net((next_s, self.critic_rec_state))
                g = g + self.gamma * v_next_s
            
        td_error = g - v_s

        # update critic using MSE(g, v(s))
        self.critic_optim.zero_grad()
        loss = td_error.pow(2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        # update the actor using Policy Gradient Theorom
        self.actor_optim.zero_grad()
        td_error = td_error.detach()  # so loss is not back-propagated to critic params
        loss = -( self.gamma_t *  td_error * self.pdf.log_prob(self.a).sum())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        self.gamma_t *= self.gamma

        for i in range(len(self.critic_rec_state)):
            self.critic_rec_state[i] = ( 
                self.critic_rec_state[i][0].detach(), 
                self.critic_rec_state[i][1].detach()
            )

