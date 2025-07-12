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

from preprocessing import preprocess_data_mark2
from environment import DayTradingEnv
from agent import DDDQNTrainer_PrioritizedReplay_BiLSTM
import os

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def main():
    # Load and preprocess data
    os.chdir("..")
    df = pd.read_csv("bitcoin_2020_2025.csv", parse_dates=True, index_col=0)
    df_processed, non_scaled_processed_df = preprocess_data_mark2(df.copy())

    # Split into train and test sets
    train_size = int(len(df_processed) * 0.7)
    train_data = df_processed[:train_size] # needs to have the same order as non_scaled_processed_df
    test_data = df_processed[train_size:]

    print(f"Training data range: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Testing data range: {test_data.index.min()} to {test_data.index.max()}")

    # Initialize training environment
    env = DayTradingEnv(train_data, non_scaled_processed_df, observation_dim='2D')

    # Test random agent behavior
    print("Environment reset:", env.reset())
    for _ in range(3):
        pact = np.random.randint(2)  # 0: buy, 1: sell
        print("Random step:", env.step(pact))

    # Train agent
    trainer = DDDQNTrainer_PrioritizedReplay_BiLSTM(env)
    trained_model = trainer.train()

    # Plot training metrics
    trainer.plot_metrics()



if __name__ == "__main__":
    main()
