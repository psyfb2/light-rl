import gym
import matplotlib.pyplot as plt

from light_rl.algorithms.ddpg import DDPG
from light_rl.common.plotting import plot_avg_reward

PENDULUM_CONFIG = {
    "env": "Pendulum-v1",
    "max_timesteps": 30000,
    "max_training_time": 10 * 60,
    "target_return": -300.0,
    "max_episode_length": 200,
    "eval_freq": 3000,
    "eval_episodes": 3,

    "gamma": 0.99,
    "actor_learning_rate": 1e-3,
    "critic_learning_rate": 1e-3,
    "critic_hidden_layers": [64, 64],
    "actor_hidden_layers": [64, 64],
    "noise_std": 0.01,
    "tau": 0.01,
    "max_grad_norm": 50,
    "batch_size": 64,
    "buffer_capacity": int(1e6),
}


def train_ddpg(config=PENDULUM_CONFIG, video_folder="ddpg_pendulum_video"):
    env =  gym.make(config["env"], new_step_api=True)

    agent = DDPG(
        env.action_space, env.observation_space,
        actor_hidden_layers=config["actor_hidden_layers"],
        critic_hidden_layers=config["critic_hidden_layers"],
        actor_lr=config["actor_learning_rate"],
        critic_lr=config["critic_learning_rate"],
        noise_std=config["noise_std"],
        gamma=config["gamma"],
        tau=config["tau"],
        max_grad_norm=config["max_grad_norm"],
        batch_size=config["batch_size"],
        buffer_capacity=config["buffer_capacity"]
    )

    rewards, times = agent.train(
        env, 
        config["max_timesteps"], 
        config["max_training_time"], 
        config["target_return"],
        config["max_episode_length"], 
        config["eval_freq"], 
        config["eval_episodes"]
    )

    # plot episodic return against episode number
    plt.plot(rewards)
    plt.title("Episodic Reward")
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.show()

    # plot episodic return against wall clock time
    plt.plot(times, rewards)
    plt.title("Episodic Reward")
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Reward")
    plt.show()

    plot_avg_reward(rewards)

    env = gym.wrappers.RecordVideo(env, video_folder, new_step_api=True)
    agent.play_episode(env, config["max_timesteps"])