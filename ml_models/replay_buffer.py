import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Experience replay buffer for DQN
    Stores transitions and samples random batches
    """
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
        
    def size(self):
        """Get current buffer size"""
        return len(self.buffer)
