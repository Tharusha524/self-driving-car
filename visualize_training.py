import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_results():
    if not os.path.exists('models/episode_rewards.npy'):
        print("Training data not found in models/ directory.")
        return

    rewards = np.load('models/episode_rewards.npy')
    lengths = np.load('models/episode_lengths.npy')

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Rewards
    ax1.plot(rewards, color='blue', alpha=0.3, label='Raw Reward')
    # Rolling average
    if len(rewards) > 10:
        rolling_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
        ax1.plot(range(9, len(rewards)), rolling_avg, color='red', linewidth=2, label='10-Ep Average')
    ax1.set_title('Training Rewards per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True)

    # Plot Episode Lengths
    ax2.plot(lengths, color='green', alpha=0.3, label='Raw Steps')
    if len(lengths) > 10:
        rolling_avg_len = np.convolve(lengths, np.ones(10)/10, mode='valid')
        ax2.plot(range(9, len(lengths)), rolling_avg_len, color='darkgreen', linewidth=2, label='10-Ep Average')
    ax2.set_title('Steps per Episode (Survival Time)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('models/training_performance.png')
    print("Graph saved to models/training_performance.png")

if __name__ == '__main__':
    plot_training_results()
