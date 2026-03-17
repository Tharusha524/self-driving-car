import sys
try:
    import pygame
except ImportError:
    import pygame_ce as pygame
    sys.modules['pygame'] = pygame

import numpy as np
import os
from environment.car import Car
from environment.track import Track
from ml_models.dqn_agent import DQNAgent
from training.config import TRAINING_CONFIG
from utils.brain_visualizer import BrainVisualizer

def test_agent(model_path, episodes=100):
    """Test trained agent with brain visualization"""
    pygame.init()
    screen = pygame.display.set_mode(
        (TRAINING_CONFIG['screen_width'], TRAINING_CONFIG['screen_height'])
    )
    pygame.display.set_caption('Self-Driving Car - Trained Agent Testing')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    # Brain Visualizer
    brain_viz = BrainVisualizer(
        TRAINING_CONFIG['simulation_width'], 0, 
        TRAINING_CONFIG['screen_width'] - TRAINING_CONFIG['simulation_width'], 
        TRAINING_CONFIG['screen_height']
    )
    
    # Load agent
    agent = DQNAgent(
        state_size=TRAINING_CONFIG['state_size'],
        action_size=TRAINING_CONFIG['action_size']
    )
    
    if os.path.exists(model_path):
        agent.load(model_path)
        agent.epsilon = 0 # Deterministic
        print(f"Model loaded: {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}. Running with random agent.")

    for episode in range(episodes):
        # Generate new random track each episode
        track = Track(TRAINING_CONFIG['simulation_width'], TRAINING_CONFIG['screen_height'], randomize=True)
        
        car = Car(400, 650, -90)
        car.max_speed = 4  # Cap speed for visualization
        step = 0
        total_reward = 0
        
        running = True
        while running and car.alive and step < 2000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            state = car.get_state()
            # Optimized: and get action + activations in one prediction call
            action, activations = agent.act(state, training=False, return_activations=True)
            
            car.update(action, track.obstacles)
            
            # Draw
            track.draw(screen)
            car.draw(screen)
            brain_viz.draw(screen, activations)
            
            # Info
            info = f'Test Ep: {episode+1} | Step: {step} | Speed: {car.speed:.2f}'
            text = font.render(info, True, (255, 255, 255))
            screen.blit(text, (20, 20))
            
            pygame.display.flip()
            clock.tick(20) # Even slower
            step += 1
            
        print(f'Episode {episode+1} completed in {step} steps.')

    pygame.quit()

if __name__ == '__main__':
    # Use final model if exists, otherwise try most recent
    model_path = 'models/dqn_car_epfinal.weights.h5'
    test_agent(model_path)
