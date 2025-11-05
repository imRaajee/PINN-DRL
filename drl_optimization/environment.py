"""
PINN environment for DRL optimization
"""

import torch
import numpy as np
from drl_config import device

class PINNEnvironment:
    """Environment for evaluating PINN configurations"""
    
    def __init__(self):
        self.state_dim = 1  # Schmidt number (Sc)
        self.action_dim = 4  # [cp1, cp2, cp3, Re]
        
    def reset(self, Sc):
        """Reset environment with given Schmidt number"""
        return torch.tensor([Sc], dtype=torch.float32)

    def step(self, actions, states):
        """
        Evaluate actions (geometry parameters) for given states (Sc)
        
        Args:
            actions: (batch_size, 4) tensor of [cp1, cp2, cp3, Re]
            states: (batch_size, 1) tensor of Schmidt numbers
            
        Returns:
            rewards: (batch_size,) tensor of efficiency values
            done: always True (single step episodes)
            info: empty dict
        """
        rewards = []
        for i in range(actions.shape[0]):
            reward = self.pinn_evaluation(actions[i], states[i])
            rewards.append(reward)
        rewards = torch.stack(rewards)
        done = torch.ones_like(rewards, dtype=torch.bool)
        return rewards, done, {}

    def calculate_mixing_index(self, concentration_matrix):
        """Calculate mixing index from concentration field"""
        mi = 1 - np.sqrt(np.sum(((concentration_matrix[:,0] - 0.5)/0.5)**2) / len(concentration_matrix))
        return mi

    def pinn_evaluation(self, action, state):
        """
        Evaluate a single configuration using the trained PINN
        
        Args:
            action: tensor of [pr1, pr2, pr3, Re]
            state: tensor of [Sc]
            
        Returns:
            efficiency: scalar reward value
        """
        pr1, pr2, pr3, Re = action
        Sc = state

        # Create input points for PINN evaluation
        # Inlet evaluation points
        xi = np.arange(0.0, 1.0, 0.1)
        yi = np.arange(1.5-0.0001, 1.5, 0.1)
        p1 = np.arange(pr1-0.0001, pr1, 0.1)
        p2 = np.arange(pr2-0.0001, pr2, 0.1)
        p3 = np.arange(pr3-0.0001, pr3, 0.1)
        re = np.arange(Re-0.0001, Re, 0.1)
        sc = np.arange(Sc-0.0001, Sc, 0.1)

        X, Y, P1, P2, P3, RE, SC = np.meshgrid(xi, yi, p1, p2, p3, re, sc)
        x = X.reshape(-1, 1)
        y = Y.reshape(-1, 1)
        p1 = P1.reshape(-1, 1)
        p2 = P2.reshape(-1, 1)
        p3 = P3.reshape(-1, 1)
        re = RE.reshape(-1, 1)
        sc = SC.reshape(-1, 1)
        XYpi = np.concatenate([x, y, p1, p2, p3, re, sc], axis=1)
        XYpi = torch.tensor(XYpi, dtype=torch.float32).to(device)

        # Outlet evaluation points
        xo = np.arange(7.0-0.0001, 7.0, 0.1)
        yo = np.arange(0.5, 1.5, 0.1)
        p1 = np.arange(pr1-0.0001, pr1, 0.1)
        p2 = np.arange(pr2-0.0001, pr2, 0.1)
        p3 = np.arange(pr3-0.0001, pr3, 0.1)
        re = np.arange(Re-0.0001, Re, 0.1)
        sc = np.arange(Sc-0.0001, Sc, 0.1)

        X, Y, P1, P2, P3, RE, SC = np.meshgrid(xo, yo, p1, p2, p3, re, sc)
        x = X.reshape(-1, 1)
        y = Y.reshape(-1, 1)
        p1 = P1.reshape(-1, 1)
        p2 = P2.reshape(-1, 1)
        p3 = P3.reshape(-1, 1)
        re = RE.reshape(-1, 1)
        sc = SC.reshape(-1, 1)
        XYpo = np.concatenate([x, y, p1, p2, p3, re, sc], axis=1)
        XYpo = torch.tensor(XYpo, dtype=torch.float32).to(device)

        # PINN evaluation (assuming 'pinn' is available)
        with torch.no_grad():
            # Note: You'll need to load your trained PINN model here
            # c = pinn.predict(XYpo)[2]  # Concentration at outlet
            # p = pinn.predict(XYpi)[6]  # Pressure at inlet
            
            # For now, using placeholder values - replace with actual PINN predictions
            c = torch.randn(XYpo.shape[0], 1)  # Placeholder
            p = torch.randn(XYpi.shape[0], 1)  # Placeholder

            c = c.cpu().numpy()
            p = p.cpu().numpy()

        # Calculate efficiency metric
        MI = self.calculate_mixing_index(c)  # Mixing index
        Cp = np.mean(p)  # Pressure cost

        efficiency = MI / (Cp ** (1/3)) if Cp > 0 else 0
        
        if np.isnan(efficiency):
            efficiency = 0

        return torch.tensor(efficiency)
