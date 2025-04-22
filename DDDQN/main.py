# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from absl import logging
logging.set_verbosity(logging.ERROR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam

from collections import deque
import gymnasium as gym
from gymnasium import spaces

from preprocessing import preprocess_data
from environment import TradingEnvironment
from agent import DDDQNTrainer

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def main():
    # Load and preprocess data
    df = pd.read_csv(r"C:\Users\ARYAN\Desktop\Trading\bitcoin_2020_2025.csv", parse_dates=True, index_col=0)
    df_processed = preprocess_data(df.copy())

    # Split into train and test sets
    train_size = int(len(df_processed) * 0.7)
    train_data = df_processed[:train_size]
    test_data = df_processed[train_size:]

    print(f"Training data range: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Testing data range: {test_data.index.min()} to {test_data.index.max()}")

    # Initialize training environment
    env = TradingEnvironment(train_data)

    # Test random agent behavior
    print("Environment reset:", env.reset())
    for _ in range(3):
        pact = np.random.randint(3)  # 0 = hold, 1 = buy, 2 = sell
        print("Random step:", env.step(pact))

    # Train agent
    trainer = DDDQNTrainer(env)
    trained_model = trainer.train()

    # Plot training metrics
    trainer.plot_metrics()

    # Evaluate on test set
    # test_env = TradingEnvironment(test_data)
   ## test_reward, test_profit, test_actions = trainer.test(test_env)

   # print(f"\nTest reward: {test_reward:.2f}")
   # print(f"Test profit: {test_profit:.2f}")
   # print(f"Test actions (first 10): {test_actions[:10]}")


if __name__ == "__main__":
    main()
