# 🚗 Self-Driving Car AI Simulation

An autonomous vehicle simulation built with **Deep Reinforcement Learning (DQN)**. The agent learns to navigate complex tracks from scratch through trial and error, moving from random guesses to expert-level driving.

![Project Preview](https://via.placeholder.com/800x400?text=Self-Driving+Car+AI+Simulation) <!-- Replace with real screenshot if available -->

## 🌟 Features
- **Deep Q-Learning (DQN)**: A neural network-based agent that optimizes driving actions (Accelerate, Turn, Brake) based on rewards.
- **LIDAR Sensor Simulation**: 5 ray-cast sensors providing real-time distance data to obstacles.
- **Custom Physics Engine**: Built with Pygame, featuring realistic acceleration, friction, and collision dynamics.
- **Brain Visualizer**: A real-time HUD showing neural network activations—watch the AI "think" as it drives!
- **Dynamic Track Generation**: Ability to test the agent on various track layouts.

## 🧠 How It Works
The car starts with zero knowledge (random weights). It learns the "Quality" (Q-value) of each action by interacting with the environment:
1. **Perception**: Read distances from 5 LIDAR sensors.
2. **Decision**: Feed data through a 4-layer Deep Neural Network.
3. **Action**: Choose the move with the highest predicted reward.
4. **Learning**: store experiences and update the network using **Experience Replay** and **Double DQN** techniques.

## 🛠️ Tech Stack
- **Language**: Python 3.11
- **Machine Learning**: TensorFlow / Keras
- **Simulation/UI**: Pygame
- **Mathematics**: NumPy
- **Visualization**: Matplotlib

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Tharusha524/self-driving-car.git
cd self-driving-car
pip install -r requirements.txt
```

### 2. Training the AI
To start the training process and watch the car learn in real-time:
```bash
python train.py
```

### 3. Testing the Agent
To run the simulation with a pre-trained model:
```bash
python test.py
```

## 📈 Learning Journey
- **Episodes 0-50**: The agent explores randomly and frequently crashes.
- **Episodes 50-200**: Learns to avoid walls and follow the track at low speeds.
- **Episodes 300+**: Masters high-speed navigation and complex cornering.

## 📄 License
This project is for educational purposes as part of a Machine Learning portfolio.

---
**Developed by [Tharusha Nanayakkara](https://github.com/Tharusha524)**
