"""
PPO agent implementation
"""

import torch
import torch.optim as optim
from torch.distributions import Normal
from networks import Actor, Critic
from drl_config import *

class PPO:
    """Proximal Policy Optimization agent"""
    
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        
        self.gamma = GAMMA
        self.eps_clip = EPS_CLIP
        self.K_epochs = K_EPOCHS
        
    def update(self, states, actions, old_log_probs, rewards, dones):
        """
        Update actor and critic networks using PPO
        
        Args:
            states: list of state tensors
            actions: list of action tensors  
            old_log_probs: list of log probability tensors
            rewards: list of reward tensors
            dones: list of done flags
        """
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Calculate advantages
        with torch.no_grad():
            values = self.critic(states).squeeze()
            advantages = rewards - values
        
        # PPO update for multiple epochs
        for _ in range(self.K_epochs):
            # Calculate new log probabilities
            means, log_stds = self.actor(states)
            stds = log_stds.exp()
            dist = Normal(means, stds)
            
            new_log_probs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().mean()
            
            # PPO ratio and surrogate loss
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            # Critic loss
            values = self.critic(states).squeeze()
            critic_loss = (rewards - values).pow(2).mean()
            
            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()