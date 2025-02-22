#!/usr/bin/env python
# coding: utf-8

"""
Reinforcement Learning: Financial Applications
Author: Dr. Yves J. Hilpisch | The Python Quants GmbH
Website: http://aimachine.io | Twitter: http://twitter.com/dyjh

This script applies Deep Q-Learning (DQL) for financial time series prediction.
It includes:
- Data processing and normalization
- Financial environment setup
- Training and testing of a Deep Q-Learning (DQL) agent

Environment: Financial Market (EUR/USD Exchange Rate)
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
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

# --- CUSTOM FINANCIAL ENVIRONMENT ---
class ObservationSpace:
    """Defines the shape of the observation space."""
    def __init__(self, n):
        self.shape = (n,)

class ActionSpace:
    """Defines the action space for the environment (Binary: Buy/Sell)."""
    def __init__(self, n):
        self.n = n

    def seed(self, seed):
        pass

    def sample(self):
        """Randomly selects an action (0 or 1)."""
        return random.randint(0, self.n - 1)

class FinanceEnv:
    """
    Custom financial environment for reinforcement learning.
    - Uses EUR/USD exchange rate data for training.
    - Defines actions (buy/sell) and corresponding rewards.
    """
    url = 'http://hilpisch.com/aiif_eikon_eod_data.csv'  # Data source

    def __init__(self, symbol='EUR=', features=['EUR=']):
        self.symbol = symbol
        self.features = features
        self.observation_space = ObservationSpace(4)  # Observing 4 time steps
        self.action_space = ActionSpace(2)  # Two possible actions: Buy (1) or Sell (0)
        self.min_accuracy = 0.475  # Minimum accuracy threshold
        self._load_data()
        self._prepare_data()

    def _load_data(self):
        """Loads historical financial data from CSV."""
        self.raw_data = pd.read_csv(self.url, index_col=0, parse_dates=True).dropna()

    def _prepare_data(self):
        """Processes financial data for RL training."""
        self.data = pd.DataFrame(self.raw_data[self.symbol])
        self.data['return'] = np.log(self.data / self.data.shift(1))  # Log returns
        self.data.dropna(inplace=True)

        # Normalize data (zero mean, unit variance)
        self.data = (self.data - self.data.mean()) / self.data.std()

        # Define the target: 1 if return > 0 (buy), 0 if return <= 0 (sell)
        self.data['direction'] = np.where(self.data['return'] > 0, 1, 0)

    def _get_state(self):
        """Returns the current state based on the last few observations."""
        return self.data[self.features].iloc[self.current_step - self.observation_space.shape[0] : self.current_step].values

    def reset(self):
        """Resets the environment for a new episode."""
        self.total_reward = 0
        self.accuracy = 0
        self.current_step = self.observation_space.shape[0]
        state = self.data[self.features].iloc[self.current_step - self.observation_space.shape[0] : self.current_step]
        return state.values, {}

    def step(self, action):
        """Executes a step in the environment based on the given action."""
        truncated = False
        correct = action == self.data['direction'].iloc[self.current_step]
        reward = 1 if correct else 0  # Reward 1 for correct prediction, 0 otherwise
        self.total_reward += reward
        self.current_step += 1
        self.accuracy = self.total_reward / (self.current_step - self.observation_space.shape[0])

        # Check if the episode is finished
        if self.current_step >= len(self.data):
            done = True
            truncated = True
        elif reward == 1:
            done = False
        elif self.accuracy < self.min_accuracy and self.current_step > self.observation_space.shape[0] + 10:
            done = True
            truncated = True
        else:
            done = False

        state = self._get_state()
        return state, reward, done, truncated, {}

# --- DEEP Q-LEARNING AGENT ---
class DQLAgent:
    """Deep Q-Learning (DQL) Agent for financial market prediction."""
    def __init__(self, gamma=0.95, lr=0.001):
        self.gamma = gamma
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.memory = deque(maxlen=2000)
        self.model = self._build_model(lr)
        self.averages = []

    def _build_model(self, lr):
        """Builds a Deep Q-Network (DQN)."""
        model = Sequential([
            Dense(24, input_dim=4, activation='relu'),
            Dense(24, activation='relu'),
            Dense(2, activation='linear')  # Two possible actions: Buy (1) or Sell (0)
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
        total_rewards = []  # Stores total rewards per episode
        for e in range(episodes):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state, info = reset_result  # Unpack correctly
            else:
                state = reset_result  # Handle older Gym versions

            state = np.reshape(state, [1, 4])
            episode_reward = 0  # Track total reward per episode

            for step in range(500):
                action = self.act(state)
                next_state, reward, done, _, _ = env.step(action)
                next_state = np.reshape(next_state, [1, 4])
                self.memory.append([state, action, reward, next_state, done])
                state = next_state
                episode_reward += reward  # Update episode total reward
                if done:
                    total_rewards.append(episode_reward)  # Store total reward for this episode
                    print(f"Episode {e+1}/{episodes}, Score: {episode_reward}")
                    break

            # Compute moving average (over the last 25 episodes)
            if len(total_rewards) >= 25:
                avg_reward = sum(total_rewards[-25:]) / 25
                self.averages.append(avg_reward)

            if len(self.memory) > self.batch_size:
                self.replay()

# --- TRAINING AND TESTING ---
# Initialize financial environment
env = FinanceEnv()

# Train DQLAgent
agent = DQLAgent()
agent.learn(100)

# Plot Training Performance
plt.figure(figsize=(10, 6))
x = range(len(agent.averages))
plt.plot(x, agent.averages, label='Moving Average')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.legend()
plt.title("Training Performance Over Episodes")
plt.show()

# Close environment
env = None
