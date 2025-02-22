#!/usr/bin/env python
# coding: utf-8

"""
Artificial Intelligence in Finance - Recurrent Neural Networks (RNNs)
Author: Dr. Yves J Hilpisch | The AI Machine
Website: http://aimachine.io | Twitter: http://twitter.com/dyjh

This script demonstrates the use of Recurrent Neural Networks (RNNs)
for financial data analysis, including time-series forecasting and classification.
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pprint import pprint
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score

# Set TensorFlow log level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'

# Set random seeds for reproducibility
def set_seeds(seed=100):
    """Set the seed for reproducibility in random operations."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()

# Generate a simple sequential dataset
a = np.arange(100).reshape((100, 1))

# Define the number of lag observations (sequence length)
lags = 3

# Create time-series generator for the dataset
g = TimeseriesGenerator(a, a, length=lags, batch_size=5)

# Display first batch from the generator
pprint(list(g)[0])

# Build a simple RNN model
model = Sequential([
    SimpleRNN(100, activation='relu', input_shape=(lags, 1)),  # RNN layer with 100 neurons
    Dense(1, activation='linear')  # Output layer (regression task)
])

# Compile the model with loss function and optimizer
model.compile(optimizer='adagrad', loss='mse', metrics=['mae'])

# Print model summary
model.summary()

# Train the model
model.fit(g, epochs=1000, steps_per_epoch=5, verbose=False)

# Evaluate model performance on sample predictions
x_sample = np.array([21, 22, 23]).reshape((1, lags, 1))
y_pred = model.predict(x_sample, verbose=False)
print(f"Predicted value: {int(round(y_pred[0, 0]))}")

# Function to generate non-linear transformed data
def transform(x):
    """
    Generate a synthetic dataset with a quadratic and sinusoidal component.
    """
    y = 0.05 * x**2 + 0.2 * x + np.sin(x) + 5
    y += np.random.standard_normal(len(x)) * 0.2  # Add noise
    return y

# Generate synthetic dataset
x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
a = transform(x).reshape((500, 1))

# Visualize the data
plt.figure(figsize=(10, 6))
plt.plot(x, a)
plt.title("Generated Non-linear Data")
plt.show()

# Define a new RNN model for this dataset
model = Sequential([
    SimpleRNN(500, activation='relu', input_shape=(lags, 1)),
    Dense(1, activation='linear')
])

# Compile and train the model
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.fit(TimeseriesGenerator(a, a, length=lags, batch_size=5), epochs=500, steps_per_epoch=10, verbose=False)

# Load financial data (EUR/USD exchange rate)
url = 'http://hilpisch.com/aiif_eikon_id_eur_usd.csv'
data = pd.read_csv(url, index_col=0, parse_dates=True)
data = data.resample('30min').last().ffill()  # Resample to 30-minute intervals

# Normalize data for better training performance
data = (data - data.mean()) / data.std()
print(data.shape)  # Debugging: Check shape before reshaping
# (4415, 4)
print(data.head(10))  # Debugging: Check first 10 rows

# p = data.values.reshape((len(data), 1))
p0 = data.values
print(p0.shape)

# Pick one column and reshape it
p1 = data.iloc[:, 0].values.reshape((-1, 1))
print(p1.shape)
# (4415, 1)

# let Python determine it size:
p2 = data.values.reshape((-1, 1))
print(p2.shape)
# (17660, 1)

# Create a generator for financial time series data
g = TimeseriesGenerator(p1, p1, length=lags, batch_size=5)

# Function to create an RNN model for financial data
def create_rnn_model(hu=100, lags=lags, layer='SimpleRNN', features=1, algorithm='estimation'):
    """
    Create and compile a simple RNN model for financial forecasting.
    """
    model = Sequential()
    if layer == 'SimpleRNN':
        model.add(SimpleRNN(hu, activation='relu', input_shape=(lags, features)))
    else:
        model.add(LSTM(hu, activation='relu', input_shape=(lags, features)))

    model.add(Dense(1, activation='linear' if algorithm == 'estimation' else 'sigmoid'))

    loss = 'mse' if algorithm == 'estimation' else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['mae' if algorithm == 'estimation' else 'accuracy'])
    return model

# Train an RNN model on financial data
model = create_rnn_model()
model.fit(g, epochs=500, steps_per_epoch=10, verbose=False)

# Predict and visualize financial time series
y_pred = model.predict(g, verbose=False)
data['pred'] = np.nan
data['pred'].iloc[lags:] = y_pred.flatten()

data.plot(figsize=(10, 6), style=['b', 'r-.'], alpha=0.75)
plt.title("Financial Time-Series Prediction")
plt.show()

# Calaulate prediction accuracy for financial data
print("calculating prediction accuracy:")

# Generate log returns for financial data
data['r'] = np.log(data['CLOSE'] / data['CLOSE'].shift(1))
data.dropna(inplace=True)
print(data.shape)
print(data.head(10))

# Normalize returns
data_one = data['r'].values.reshape((len(data), 1))
print(data_one.shape)

# Create generator for return data
g = TimeseriesGenerator(data_one, data_one, length=lags, batch_size=5)

# Train a model on financial return data
model = create_rnn_model()
model.fit(g, epochs=500, steps_per_epoch=10, verbose=False)

# Compute prediction accuracy
y_pred = model.predict(g, verbose=False)
data['pred'] = np.nan
data['pred'].iloc[lags:] = y_pred.flatten()
data.dropna(inplace=True)

accuracy = accuracy_score(np.sign(data['r']), np.sign(data['pred']))
print(f"Prediction accuracy: {accuracy:.4f}")
