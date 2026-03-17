"""Quick headless runner: runs a few episodes using a random agent.
This avoids installing `pygame` and `tensorflow` and runs a short simulation
to validate the environment and basic loop.
"""
import random
import numpy as np
from environment.car import Car
from environment.track import Track
from training.config import TRAINING_CONFIG


class RandomAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.0

    def act(self, state, training=True, return_activations=False):
        a = random.randrange(self.action_size)
        if return_activations:
            return a, None
        return a

    def remember(self, *args, **kwargs):
        pass

    def replay(self):
        return 0

    def save(self, path):
        pass


def calculate_reward(car, track, config):
    reward = 0
    if not car.alive:
        reward += config['penalty_collision']
    else:
        reward += config['reward_alive']
        reward += car.speed * config['reward_distance']
        if car.speed < 1:
            reward += config['penalty_slow']
    return reward


def run(episodes=3, max_steps=500):
    cfg = TRAINING_CONFIG.copy()
    cfg['episodes'] = episodes
    cfg['max_steps'] = max_steps

    track = Track(cfg['simulation_width'], cfg['screen_height'])
    agent = RandomAgent(cfg['state_size'], cfg['action_size'])

    results = []
    for ep in range(episodes):
        car = Car(400, 650, -90)
        total_reward = 0
        step = 0
        for step in range(max_steps):
            state = car.get_state()
            action = agent.act(state, training=False)
            car.update(action, track.obstacles)
            r = calculate_reward(car, track, cfg)
            total_reward += r
            agent.remember(state, action, r, car.get_state(), not car.alive)
            _ = agent.replay()
            if not car.alive:
                break
        results.append((ep, step + 1, total_reward))
        print(f'Episode {ep+1}: steps={step+1}, reward={total_reward:.2f}')

    print('\nSummary:')
    for ep, steps, reward in results:
        print(f' Ep {ep+1}: steps={steps}, reward={reward:.2f}')


if __name__ == '__main__':
    run(episodes=3, max_steps=500)
