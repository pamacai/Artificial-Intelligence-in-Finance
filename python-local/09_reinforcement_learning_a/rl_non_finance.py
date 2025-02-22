#!/usr/bin/env python
# coding: utf-8

"""
Reinforcement Learning: Non-Financial Applications (CartPole)
Author: Dr. Yves J. Hilpisch | The Python Quants GmbH
Website: http://aimachine.io | Twitter: http://twitter.com/dyjh

This script demonstrates reinforcement learning using:
- A simple weight-based policy
- A neural network agent (NNAgent)
- A deep Q-learning (DQL) agent

Environment: OpenAI Gym CartPole-v0
"""

import os
import math
import random
import numpy as np
import pandas as pd
import gym
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from collections import deque
import matplotlib.pyplot as plt

# Set environment variables to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'

# Fix randomness for reproducibility
def set_seeds(seed=100):
    """Set seeds for reproducibility in random operations."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()

# Initialize CartPole environment
env = gym.make('CartPole-v0')
env.action_space.seed(100)

# --- SIMPLE WEIGHT-BASED POLICY ---
def run_episode(env, weights):
    """Runs one episode using a simple weight-based policy."""
    state = env.reset()[0]
    total_reward = 0
    for _ in range(200):
        s = np.dot(state, weights)  # Weighted sum of state
        action = 0 if s < 0 else 1  # Action decision based on sign
        state, reward, done, trunc, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

# Generate random weights
weights = np.random.random(4) * 2 - 1
print("Sample Reward with Random Weights:", run_episode(env, weights))

# --- NEURAL NETWORK AGENT ---
class NNAgent:
    """Neural Network-based agent for reinforcement learning."""
    def __init__(self):
        self.scores = []  # Store scores of each episode
        self.model = self._build_model()

    def _build_model(self):
        """Builds a simple neural network model."""
        model = Sequential([
            Dense(24, input_dim=4, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.legacy.RMSprop(learning_rate=0.001))
        return model

    def act(self, state):
        """Selects an action based on model prediction or randomness."""
        if random.random() <= 0.5:
            return env.action_space.sample()
        return np.where(self.model.predict(state)[0, 0] > 0.5, 1, 0)

    def train(self, episodes):
        """Trains the model using reinforcement learning."""
        for e in range(episodes):
            state = env.reset()[0]
            for _ in range(201):
                state = np.reshape(state, [1, 4])
                action = self.act(state)
                next_state, reward, done, trunc, info = env.step(action)
                if done:
                    self.scores.append(_ + 1)
                    break
                self.model.fit(state, np.array([action]), epochs=1, verbose=False)
                state = next_state

# Train NNAgent
agent = NNAgent()
agent.train(1000)

# --- DEEP Q-LEARNING AGENT ---
class DQLAgent:
    """Deep Q-Learning (DQL) Agent using a Neural Network."""
    def __init__(self, gamma=0.95, lr=0.001):
        self.gamma = gamma
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.memory = deque(maxlen=2000)
        self.model = self._build_model(lr)

    def _build_model(self, lr):
        """Builds a Deep Q-Network (DQN)."""
        model = Sequential([
            Dense(24, input_dim=4, activation='relu'),
            Dense(24, activation='relu'),
            Dense(env.action_space.n, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.legacy.Adam(learning_rate=lr))
        return model

    def act(self, state):
        """Selects an action using an ε-greedy strategy."""
        if random.random() <= self.epsilon:
            return env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def replay(self):
        """Trains the network using experience replay."""
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                reward += self.gamma * np.amax(self.model.predict(next_state)[0])
            target = self.model.predict(state)
            target[0, action] = reward
            self.model.fit(state, target, epochs=1, verbose=False)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, episodes):
        """Main training loop."""
        for e in range(episodes):
            state = env.reset()[0]
            state = np.reshape(state, [1, 4])
            for _ in range(500):
                action = self.act(state)
                next_state, reward, done, trunc, info = env.step(action)
                next_state = np.reshape(next_state, [1, 4])
                self.memory.append([state, action, reward, next_state, done])
                state = next_state
                if done:
                    print(f"Episode {e+1}/{episodes}, Score: {_ + 1}")
                    break
            if len(self.memory) > self.batch_size:
                self.replay()

# Train DQLAgent
dql_agent = DQLAgent()
dql_agent.learn(1000)

# Close environment
env.close()
