# Training configuration
TRAINING_CONFIG = {
    # Environment
    'screen_width': 1200,  # Wider to accommodate brain visualization
    'screen_height': 700,
    'simulation_width': 800,
    'fps': 60,
    
    # Agent
    'state_size': 8,  # 5 sensors + speed + sin(angle) + cos(angle)
    'action_size': 5,  # nothing, forward, left, right, brake
    
    # Training
    'episodes': 500,
    'max_steps': 1500, # More time to finish the track
    'batch_size': 128, # Larger batch for more stable gradients
    'epsilon_decay': 0.993, 
    
    # Rewards
    'reward_alive': 0.5, # Stronger survival reward
    'reward_checkpoint': 100.0,
    'reward_distance': 0.2,
    'penalty_collision': -100, # Less destructive penalty
    'penalty_slow': -0.1,
    
    # Model saving
    'save_interval': 50,
    'model_path': 'models/dqn_car.weights.h5'
}
