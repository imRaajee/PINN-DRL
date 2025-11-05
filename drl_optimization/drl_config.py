"""
Configuration for DRL-PPO optimization
"""

import torch
import numpy as np

# Device configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Action bounds for normalization
ACTION_BOUNDS = {
    'cp': [0.0, 1.0],  # Control points bounds
    'Re': [5, 80]      # Reynolds number bounds
}

# Training parameters
BATCH_SIZE = 64
TOTAL_EPISODES = 100
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 10
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3

# Schmidt number range for training
SC_MIN = 0.1
SC_MAX = 10.0