"""
Main training script for PPO optimization
"""

import torch
import numpy as np
from environment import PINNEnvironment
from ppo_agent import PPO
from visualization import create_visualizations
from drl_config import *

def train_ppo():
    """Main training loop for PPO optimization"""
    env = PINNEnvironment()
    agent = PPO(env.state_dim, env.action_dim)
    
    rewards_history = []
    
    # Store final batch for visualization
    final_states = None
    final_actions = None
    final_rewards = None
    
    print("Starting PPO training for PINN optimization...")
    
    for episode in range(TOTAL_EPISODES):
        # Sample batch of Sc values
        Sc_batch = np.random.uniform(SC_MIN, SC_MAX, size=(BATCH_SIZE, 1))
        state_batch = torch.tensor(Sc_batch, dtype=torch.float32)
        
        # Get actions from the actor
        with torch.no_grad():
            mean, log_std = agent.actor(state_batch)
            std = log_std.exp()
            dist = Normal(mean, std)
            action_batch = dist.sample()
            log_prob_batch = dist.log_prob(action_batch).sum(dim=1)
            
            # Scale actions to physical ranges
            scaled_action_batch = action_batch.clone()
            scaled_action_batch[:, :3] = scaled_action_batch[:, :3] * (ACTION_BOUNDS['cp'][1] - ACTION_BOUNDS['cp'][0]) / 2
            scaled_action_batch[:, 3] = (
                scaled_action_batch[:, 3] * (ACTION_BOUNDS['Re'][1] - ACTION_BOUNDS['Re'][0]) / 2
                + (ACTION_BOUNDS['Re'][1] + ACTION_BOUNDS['Re'][0]) / 2
            )
        
        # Evaluate actions using PINN environment
        rewards, done, _ = env.step(scaled_action_batch, state_batch)
        
        # Update PPO agent
        agent.update(
            states=[s for s in state_batch],
            actions=[a for a in action_batch],
            old_log_probs=[lp for lp in log_prob_batch],
            rewards=[r for r in rewards],
            dones=[d for d in done],
        )
        
        # Store final batch data for visualization
        if episode == TOTAL_EPISODES - 1:
            final_states = Sc_batch
            final_actions = scaled_action_batch.numpy()
            final_rewards = rewards.numpy()
        
        mean_reward = rewards.mean().item()
        rewards_history.append(mean_reward)
        
        # Print progress
        if episode % (TOTAL_EPISODES // 10) == 0:
            print(f"Episode {episode}, Avg Reward: {mean_reward:.4f}")
    
    # Save trained models
    torch.save(agent.actor.state_dict(), "ppo_actor.pth")
    torch.save(agent.critic.state_dict(), "ppo_critic.pth")
    print("Models saved: ppo_actor.pth, ppo_critic.pth")
    
    # Create comprehensive visualizations
    create_visualizations(rewards_history, final_states, final_actions, final_rewards)
    print("Training analysis saved: training_analysis.png")
    
    return rewards_history

if __name__ == "__main__":
    rewards = train_ppo()
    print("PPO training completed!")