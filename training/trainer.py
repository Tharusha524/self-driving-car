try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    try:
        import pygame_ce as pygame
        import sys as _sys
        _sys.modules['pygame'] = pygame
        PYGAME_AVAILABLE = True
    except ImportError:
        pygame = None
        PYGAME_AVAILABLE = False

import numpy as np
import sys
import os

from environment.car import Car
from environment.track import Track
from ml_models.dqn_agent import DQNAgent
from training.config import TRAINING_CONFIG
from utils.brain_visualizer import BrainVisualizer

class Trainer:
    """
    Training loop for self-driving car with visual brain UI.
    """
    def __init__(self, config=TRAINING_CONFIG, render=True):
        self.config = config
        self.render_enabled = render
        
        # Initialize Pygame if available and rendering requested
        if self.render_enabled and PYGAME_AVAILABLE:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (config['screen_width'], config['screen_height'])
            )
            pygame.display.set_caption('Self-Driving Car AI - Professional ML Simulation')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 28)
            
            # Brain Visualizer (placed in the side panel)
            self.brain_viz = BrainVisualizer(
                config['simulation_width'], 0, 
                config['screen_width'] - config['simulation_width'], 
                config['screen_height']
            )
        else:
            if self.render_enabled and not PYGAME_AVAILABLE:
                print('Warning: pygame is not available. Running without rendering.')
            self.render_enabled = False
        
        # Initialize environment
        self.track = Track(config['simulation_width'], config['screen_height'])
        
        # Initialize agent
        self.agent = DQNAgent(
            state_size=config['state_size'],
            action_size=config['action_size']
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
        # Checkpoints tracking
        self.checkpoints_reached = set()
        
    def calculate_reward(self, car, action):
        """
        Calculate refined reward for state
        """
        reward = 0
        
        if not car.alive:
            reward = self.config['penalty_collision']
        else:
            # 1. Base Survival Reward
            reward += self.config['reward_alive']
            
            # 2. Progress Reward (Speed/Distance)
            # Higher reward for maintaining good speed
            speed_ratio = car.speed / car.max_speed
            reward += speed_ratio * self.config['reward_distance']
            
            # 3. Center Lane Reward (Strongly Weighted)
            # sensor_readings: [L60, L30, C, R30, R60]
            # Perfect balance is when L30 == R30 and L60 == R60
            left_side = (car.sensor_readings[0] + car.sensor_readings[1])
            right_side = (car.sensor_readings[3] + car.sensor_readings[4])
            balance = 1.0 - abs(left_side - right_side) / 2.0
            
            # We want balance to be 1.0. Reward the agent for high balance.
            reward += balance * 0.5 
            
            # 4. Proximity Penalty (Soft Collision Avoidance)
            min_dist = min(car.sensor_readings)
            if min_dist < 0.2:
                reward -= (1.2 - min_dist) # Scale penalty as it gets closer
            
            # 5. Checkpoint Rewards
            car_rect = car.get_rect()
            for i, checkpoint in enumerate(self.track.checkpoints):
                if car_rect.colliderect(checkpoint) and i not in self.checkpoints_reached:
                    self.checkpoints_reached.add(i)
                    reward += self.config['reward_checkpoint']
                    
            if car.speed < 1:
                reward += self.config['penalty_slow']
                
        return reward
        
    def train_episode(self, episode):
        """Train one episode"""
        car = Car(400, 650, -90) # Start at bottom of the lane, facing up
        self.checkpoints_reached = set()
        
        total_reward = 0
        step = 0
        
        for step in range(self.config['max_steps']):
            if self.render_enabled:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                        
            state = car.get_state()
            action = self.agent.act(state, training=True)
            
            # Get brain activations for visualization
            activations = self.agent.get_brain_data(state)
            
            car.update(action, self.track.obstacles)
            reward = self.calculate_reward(car, action)
            total_reward += reward
            
            next_state = car.get_state()
            done = not car.alive
            
            self.agent.remember(state, action, reward, next_state, done)
            
            # Replay (train) every step
            loss = self.agent.replay()
            if loss > 0:
                self.losses.append(loss)
            
            if self.render_enabled:
                self.render(car, episode, step, total_reward, activations)
                
            if done:
                break
                
        return total_reward, step
        
    def render(self, car, episode, step, reward, activations):
        """Render the environment and brain visualization with a polished HUD"""
        # Draw Environment
        self.track.draw(self.screen)
        car.draw(self.screen)
        
        # Draw Brain UI Panel
        self.brain_viz.draw(self.screen, activations)
        
        # --- Sleek Game HUD ---
        hud_width = 280
        hud_height = 260
        hud_surface = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        # Gradient-like background
        pygame.draw.rect(hud_surface, (0, 0, 0, 180), (0, 0, hud_width, hud_height), border_radius=15)
        pygame.draw.rect(hud_surface, (0, 255, 150, 255), (0, 0, hud_width, hud_height), width=2, border_radius=15)
        
        # Title
        title_font = pygame.font.Font(None, 32)
        title_surf = title_font.render("AUTONOMOUS PILOT", True, (0, 255, 150))
        hud_surface.blit(title_surf, (hud_width//2 - title_surf.get_width()//2, 15))
        
        pygame.draw.line(hud_surface, (0, 255, 150, 100), (20, 45), (hud_width-20, 45), 1)

        info_texts = [
            f'EPISODE: {episode}',
            f'STEP: {step}',
            f'REWARD: {reward:.1f}',
            f'SENSORS: ACTIVE',
            f'SPEED: {car.speed:.2f} m/s',
            f'PROGRESS: {len(self.checkpoints_reached)}/{len(self.track.checkpoints)}'
        ]
        
        y_offset = 65
        for text in info_texts:
            text_surface = self.font.render(text, True, (220, 220, 220))
            hud_surface.blit(text_surface, (25, y_offset))
            y_offset += 30
            
        self.screen.blit(hud_surface, (20, 20))
            
        pygame.display.flip()
        self.clock.tick(self.config['fps'])
        
    def train(self):
        """Main training loop"""
        print(f"Starting Training: {self.config['episodes']} episodes")
        
        for episode in range(self.config['episodes']):
            result = self.train_episode(episode)
            if result is None: break
                
            total_reward, steps = result
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Progress print
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f'Ep {episode} | Steps: {steps} | Reward: {total_reward:.1f} | Avg (10): {avg_reward:.1f} | Epsilon: {self.agent.epsilon:.3f}')
            
            # Decay epsilon at the end of each episode
            self.agent.decay_epsilon()
            
            if episode % self.config['save_interval'] == 0 and episode > 0:
                self.save_model(episode)
                
        self.save_model('final')
        if self.render_enabled: pygame.quit()
        print("Training completed!")
        
    def save_model(self, episode):
        os.makedirs('models', exist_ok=True)
        filepath = f'models/dqn_car_ep{episode}.weights.h5'
        self.agent.save(filepath)
        np.save('models/episode_rewards.npy', np.array(self.episode_rewards))
        np.save('models/episode_lengths.npy', np.array(self.episode_lengths))
