import gym
import numpy as np
import matplotlib.pyplot as plt
import torch.nn
import torch.multiprocessing as mp
import os

from torch.optim import Adam
from torch.distributions.normal import Normal
from gym import wrappers
from tqdm import tqdm
from copy import deepcopy

from agent import Agent
from plotting import plot_avg_reward
from fc_network import FCNetwork
from continious_pg_mountaincar import Transform, ActorCritic, RBF_Transform


class SharedAdam(Adam):
    """Implements Adam algorithm with shared states.
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1) 
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) 

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * np.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class A3C(ActorCritic):
    def __init__(self, env: gym.Env, ft_transformer: Transform, 
            actor_hidden_layers=[], critic_hidden_layers=[], 
            actor_lr=1e-4, critic_lr=1e-4, gamma=1.0,
            shared_optimizer=True):
        """
        Async Actor Critic agent, assumes continious actions.
        """
        super().__init__(env, ft_transformer, 
            actor_hidden_layers, critic_hidden_layers, 
            actor_lr, critic_lr, gamma)
        
        self.actor_hidden_layers = actor_hidden_layers
        self.critic_hidden_layers = critic_hidden_layers

        if shared_optimizer:
            self.actor_optim = SharedAdam(
                self.actor_net.parameters(), lr=actor_lr
            )
            self.actor_optim.share_memory()

            self.critic_optim = SharedAdam(
                self.critic_net.parameters(), lr=critic_lr
            )
            self.critic_optim.share_memory()

        # put the actor and critic params in shared memory
        # (grads are None and so won't be put in shared memory)        
        self.actor_net.share_memory()
        self.critic_net.share_memory()
    
    def _ensure_shared_grads(self, model, shared_model):
        for param, shared_param in zip(model.parameters(),
                                       shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
    
    def _a3c_worker(self, rank: int, n_episodes: int, t_max: int,
            max_episode_length: int, entropy_beta: float, max_grad_norm: float,
            episode_counter: mp.Value, lock: mp.Lock, return_q: mp.Queue):
        env = deepcopy(self.env)

        local_actor_net = FCNetwork(
            (self.state_dim, *self.actor_hidden_layers, self.action_dim * 2), 
        )
        local_critic_net = FCNetwork(
            (self.state_dim, *self.critic_hidden_layers, 1)
        )

        s = self._transform_state(env.reset())
        episode_reward = episode_length = 0
        episode_rewards = []

        while True:
            with lock:
                if episode_counter.value >= n_episodes - 1:
                    break

            # sync local nets with global nets
            local_actor_net.load_state_dict(self.actor_net.state_dict())
            local_critic_net.load_state_dict(self.critic_net.state_dict())

            states = []    # s_t            for t in {t_start, ..., t_start + t_max - 1}
            log_probs = [] # log_prob(a_t)  for t in {t_start, ..., t_start + t_max - 1}
            entropies = [] # entropy(a_t)   for t in {t_start, ..., t_start + t_max - 1}
            rewards = []   # r_t            for t in {t_start + 1, ..., t_start + t_max}

            for step in range(t_max):
                # get action
                actor_out = local_actor_net(s) # shape => (2 * self.action_dim, )
                mu = actor_out[:self.action_dim] # shape => (self.action_dim, )
                std = self.soft_plus(actor_out[self.action_dim:]) # shape => (self.action_dim, )
                pdf = Normal(mu, std)
                a = pdf.sample()  # shape => (self.action_dim, )

                next_s, r, terminal, truncated, _ = env.step(a.detach().numpy())
                next_s = self._transform_state(next_s)

                log_probs.append(pdf.log_prob(a).sum())
                entropies.append(entropy_beta * torch.log(torch.prod(std)))
                states.append(s)
                rewards.append(r)

                episode_reward += r
                episode_length += 1
                s = next_s

                if terminal or episode_length >= max_episode_length:
                    episode_rewards.append(episode_reward)
                    s = self._transform_state(env.reset())
                    episode_reward = episode_length = 0
                    with lock:
                        episode_counter.value += 1
                        print(episode_counter.value, episode_rewards[-1])
                    break
            
            with torch.no_grad():
                R = 0 if terminal else local_critic_net(next_s) # bootstrap for n-step reward

            actor_loss = critic_loss = 0

            for i in range(len(states) - 1, -1, -1):
                R = rewards[i] + self.gamma * R
                td_error = R - local_critic_net(states[i])

                critic_loss += td_error.pow(2).sum()
                actor_loss -= td_error.detach() * log_probs[i] + entropies[i]
        
            # update global nets with accumulated gradients
            self.critic_optim.zero_grad()
            self.actor_optim.zero_grad()

            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_critic_net.parameters(), max_grad_norm)

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_actor_net.parameters(), max_grad_norm)

            self._ensure_shared_grads(local_actor_net, self.actor_net)
            self._ensure_shared_grads(local_critic_net, self.critic_net)

            self.critic_optim.step()
            self.actor_optim.step()
        
        return_q.put((rank, episode_rewards))
    
    def train_agent(self, num_processes=os.cpu_count(), n_episodes=150, t_max=20,
            max_episode_length=float("inf"), entropy_beta=1e-5, max_grad_norm=50) -> list:
        episode_counter = mp.Value('i', 0)
        return_q = mp.Queue()
        lock = mp.Lock()
        processes = []

        for rank in range(num_processes):
            p = mp.Process(target=self._a3c_worker, args=(
                    rank, n_episodes, t_max, max_episode_length, entropy_beta, max_grad_norm, 
                    episode_counter, lock, return_q
                )
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        episode_rewards = []
        for _ in range(num_processes):
            episode_rewards.append(return_q.get())
        episode_rewards.sort(key=lambda tpl : tpl[0])
        episode_rewards = [tpl[1] for tpl in episode_rewards]

        # list of list, each inner list episode reward overtime for worker
        return episode_rewards  


def play_episode(env: gym.Env, agent: Agent):
    done = False
    agent.reset()
    s = env.reset()
    episode_reward = 0
    iters = 0

    while not done and iters < 2000:
        a = agent.get_action(s)
        next_s, r, terminal, truncated, info = env.step(a)
        done = terminal

        s = next_s
        episode_reward += r
        iters += 1
    
    return episode_reward

def main(n_episodes=50):
    env =  gym.make('MountainCarContinuous-v0', new_step_api=True)

    import pickle
    with open('rbf_transformer.pkl', 'rb') as handle:
        rbf = pickle.load(handle)

    agent = A3C(env, rbf, [], [], gamma=0.9999)
    worker_rewards = agent.train_agent(4, n_episodes=n_episodes, max_episode_length=2000, t_max=10, entropy_beta=1e-5)
    
    for rewards in worker_rewards:
        plt.plot(rewards)
        plt.show()
        plot_avg_reward(rewards)

    env = wrappers.RecordVideo(env, 'a3c_car_vids', new_step_api=True)
    play_episode(env, agent)


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    main()