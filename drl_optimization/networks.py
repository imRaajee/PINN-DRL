"""
Neural networks for PPO agent
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

class Actor(nn.Module):
    """Policy network for PPO"""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        
        self.mean = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Sigmoid()  # Output between 0 and 1
        )
        self.log_std_layer = nn.Linear(64, action_dim)
    
    def forward(self, state):
        features = self.net(state)
        mean = self.mean(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -5, 0)  # Bound standard deviation
        return mean, log_std

class Critic(nn.Module):
    """Value network for PPO"""
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    
    def forward(self, state):
        return self.net(state)