import numpy as np
import matplotlib.pyplot as plt


def plot_avg_reward(rewards: np.ndarray):
    plt.plot([np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))])
    plt.title("Average Reward")
    plt.xlabel("Episode Number")
    plt.ylabel("Average Reward")
    plt.show()