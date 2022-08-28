import os
import gym
import matplotlib.pyplot as plt

from light_rl.algorithms.es import ES
from light_rl.common.plotting import plot_avg_reward

PENDULUM_CONFIG = {
    "env": "Pendulum-v1",
    "max_timesteps": 1e7,
    "max_training_time": 15 * 60,
    "target_return": -300.0,
    "max_episode_length": 200,
    "eval_freq": 100000,
    "eval_episodes": 3,

    "actor_hidden_layers": [64, 64],
    "lstm_hidden_dim": 256,
    "lr": 0.1,
    "pop_size": 50,
    "std_noise": 0.5,
    "vbn_states": False,
    "num_workers": os.cpu_count(),
    "reward_shaping_method": "rank"
}


def train_es(config=PENDULUM_CONFIG, video_folder=os.path.join("videos", "es_pendulum")):
    env =  gym.make(config["env"], new_step_api=True, render_mode="single_rgb_array")

    agent = ES(
        env.action_space, env.observation_space,
        actor_hidden_layers=config["actor_hidden_layers"],
        lstm_hidden_dim=config["lstm_hidden_dim"],
        lr=config["lr"],
        num_workers=config["num_workers"],
        pop_size=config["pop_size"],
        std_noise=config["std_noise"],
        vbn_states= ES.sample_vbn_states(env) if config["vbn_states"] else None,
        reward_shaping_method=config["reward_shaping_method"]
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

    # record video hangs for some reason here
    # video_env = gym.wrappers.RecordVideo(env, video_folder, new_step_api=True)
    # agent.play_episode(video_env, config["max_timesteps"])

    # agents can be saved and loaded for later use
    agent.save(video_folder)
    del agent
    agent = ES.load(video_folder)
    # play episode, could also use your own episode loop using agent.get_action
    reward, iters = agent.play_episode(env, config["max_episode_length"])
    print("Agent reward after saving and reloading:", reward)