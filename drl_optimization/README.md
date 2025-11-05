# DRL-PPO Optimization for PINN

This module uses Deep Reinforcement Learning (Proximal Policy Optimization) to optimize geometry parameters for the PINN-based mixer design.

## File Structure

- `drl_config.py` - Configuration parameters and constants
- `networks.py` - Actor and Critic neural networks for PPO
- `environment.py` - PINN evaluation environment
- `ppo_agent.py` - PPO algorithm implementation
- `visualization.py` - Training analysis and plotting utilities
- `train_ppo.py` - Main training script

## Usage

1. Ensure you have a trained PINN model available
2. Run the training:
   ```bash
   python train_ppo.py
Outputs
ppo_actor.pth - Trained actor network

ppo_critic.pth - Trained critic network

training_analysis.png - Comprehensive training analysis
