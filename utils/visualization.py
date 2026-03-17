import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_metrics():
    """Plot training progress from saved .npy files"""
    rewards_path = 'models/episode_rewards.npy'
    lengths_path = 'models/episode_lengths.npy'
    
    if not os.path.exists(rewards_path):
        print("Error: Training metrics not found. Run training first.")
        return

    rewards = np.load(rewards_path)
    lengths = np.load(lengths_path)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Rewards plot
    axes[0].plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(rewards) >= 50:
        window = 50
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), moving_avg, color='red', linewidth=2, label='Moving Avg (50)')
    
    axes[0].set_title('Training Rewards over Episodes')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Episode lengths plot
    axes[1].plot(lengths, alpha=0.3, color='green', label='Steps Survived')
    if len(lengths) >= 50:
        window = 50
        moving_avg_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(lengths)), moving_avg_len, color='darkgreen', linewidth=2, label='Moving Avg (50)')
        
    axes[1].set_title('Episode Duration (Steps)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_performance.png')
    plt.show()
    print("Metrics visualization saved to models/training_performance.png")

if __name__ == '__main__':
    plot_training_metrics()
