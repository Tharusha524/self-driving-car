import numpy as np
import random
from .neural_network import DQNNetwork
from .replay_buffer import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Network Agent with epsilon-greedy exploration
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.993 # Slower decay for 500 episodes
        self.learning_rate = 0.0005 # Slower learning rate for stability
        self.batch_size = 64
        self.tau = 0.005 # Soft update parameter
        
        # Networks (Double DQN)
        self.model = DQNNetwork(state_size, action_size, self.learning_rate)
        self.target_model = DQNNetwork(state_size, action_size, self.learning_rate)
        self.update_target_model()
        
        # Replay buffer
        self.memory = ReplayBuffer(max_size=20000)
        
        # Training metrics
        self.training_step = 0
        
    def update_target_model(self):
        """Copy weights to target network"""
        self.target_model.model.set_weights(self.model.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
        
    def act(self, state, training=True, return_activations=False):
        """
        Choose action using epsilon-greedy policy.
        Optionally returns activations for visualization to save performance.
        """
        state_reshaped = np.reshape(state, [1, self.state_size])
        
        if training and random.random() < self.epsilon:
            # Explore: random action
            action = random.randrange(self.action_size)
            activations = self.get_brain_data(state) if return_activations else None
            return (action, activations) if return_activations else action
        
        # Exploit: choose best action
        if return_activations:
            # Get everything in one go for efficiency
            activations = self.model.get_activations(state_reshaped)
            # Output layer is the last one
            q_values = activations[-1]
            return np.argmax(q_values[0]), activations
        else:
            q_values = self.model.predict(state_reshaped)
            return np.argmax(q_values[0])
        
    def replay(self):
        """
        Train on batch from replay buffer (Experience Replay)
        """
        if self.memory.size() < self.batch_size:
            return 0
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Predict Q-values for current states
        current_q = self.model.predict(states)
        
        # Predict Q-values for next states (using target network)
        next_q = self.target_model.predict(next_states)
        
        # Update Q-values with Bellman equation
        for i in range(self.batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Trained model
        loss = self.model.train_on_batch(states, current_q)
        
        self.training_step += 1
        
        # Update target network periodically (Soft Update)
        self.soft_update_target_model()
            
        return loss.history['loss'][0] if hasattr(loss, 'history') else 0

    def soft_update_target_model(self):
        """Soft update model weights: target = tau * model + (1 - tau) * target"""
        weights = self.model.model.get_weights()
        target_weights = self.target_model.model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.model.set_weights(target_weights)

    def decay_epsilon(self):
        """Decay epsilon at the end of each episode"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def get_brain_data(self, state):
        """
        Returns activations for visualization.
        """
        state_reshaped = np.reshape(state, [1, self.state_size])
        return self.model.get_activations(state_reshaped)
        
    def save(self, filepath):
        """Save agent weights"""
        self.model.save(filepath)
        
    def load(self, filepath):
        """Load agent weights"""
        self.model.load(filepath)
        self.update_target_model()
