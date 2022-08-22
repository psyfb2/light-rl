import gym
import matplotlib.pyplot as plt

from light_rl.algorithms.vanilla_policy_gradient import VanillaPolicyGradient
from light_rl.common.plotting import plot_avg_reward
from light_rl.common.transforms.rbf import RBFTransform

CONTINIOUS_MOUNTAIN_CAR_CONFIG = {
    "env": "MountainCarContinuous-v0",
    "max_timesteps": 30000,
    "max_training_time": 10 * 60,
    "target_return": 90,
    "max_episode_length": 2000,
    "eval_freq": 3000,
    "eval_episodes": 10,

    "gamma": 0.9999,
    "actor_learning_rate": 1e-4,
    "critic_learning_rate": 1e-4,
    "actor_adam_eps": 1e-3,
    "critic_adam_eps": 1e-3,
    "critic_hidden_layers": [],
    "actor_hidden_layers": ['lstm'],
    "max_grad_norm": 50,

    "rbf_n_episodes": 250,
    "rbf_n_components": 200,
    "rbf_gammas": (0.05, 0.1, 0.5, 1.0)
}


def train_vanilla_pg(config=CONTINIOUS_MOUNTAIN_CAR_CONFIG, video_folder="vanilla_pg_continious_mc_video"):
    env =  gym.make(config["env"], new_step_api=True)

    agent = VanillaPolicyGradient(
        env.action_space, env.observation_space, 
        RBFTransform(
            env,
            config["rbf_n_episodes"],
            config["rbf_n_components"],
            config["rbf_gammas"]
        ),
        actor_hidden_layers=config["actor_hidden_layers"],
        critic_hidden_layers=config["critic_hidden_layers"],
        actor_lr=config["actor_learning_rate"],
        critic_lr=config["critic_learning_rate"],
        actor_adam_eps=config["actor_adam_eps"],
        critic_adam_eps=config["critic_adam_eps"],
        gamma=config["gamma"],
        max_grad_norm=config["max_grad_norm"],
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