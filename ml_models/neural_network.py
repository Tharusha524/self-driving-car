import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DQNNetwork:
    """
    Deep Q-Network for decision making
    Input: sensor readings + speed + angle
    Output: Q-values for each action
    """
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        self.model = self.build_model()
        
    def build_model(self):
        """
        Build neural network architecture
        Using Dense layers for Q-value approximation
        """
        model = keras.Sequential([
            # Input layer
            layers.Dense(256, activation='relu', input_shape=(self.state_size,)),
            
            # Hidden layers
            layers.Dense(512, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            
            # Output layer (Q-values for each action)
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
        
    def predict(self, state):
        """Predict Q-values for a state"""
        return self.model.predict(state, verbose=0)
        
    def train_on_batch(self, states, targets):
        """Train on a batch of experiences"""
        return self.model.fit(states, targets, epochs=1, verbose=0)
        
    def save(self, filepath):
        """Save model weights"""
        self.model.save_weights(filepath)
        
    def load(self, filepath):
        """Load model weights"""
        self.model.load_weights(filepath)
        
    def get_activations(self, state):
        """
        Special helper for the visual brain UI.
        Returns activations for all layers given an input state.
        """
        activations = []
        current_input = state
        for layer in self.model.layers:
            current_input = layer(current_input)
            # Only track activations of Dense layers for visualization
            if isinstance(layer, layers.Dense):
                activations.append(current_input.numpy())
        return activations
