"""
PINN evaluation for genetic algorithm fitness calculation
"""

import torch
import numpy as np
from ga_config import NUM_EVAL_POINTS

class PINNEvaluator:
    """Evaluates individual solutions using the trained PINN"""
    
    def __init__(self):
        # Note: You'll need to load your trained PINN model here
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
    def calculate_mixing_index(self, concentration_matrix):
        """Calculate mixing index from concentration field"""
        if len(concentration_matrix) == 0:
            return 0.0
        mi = 1 - np.sqrt(np.sum(((concentration_matrix[:, 0] - 0.5) / 0.5) ** 2) / len(concentration_matrix))
        return np.clip(mi, 0.0, 1.0)
    
    def evaluate_individual(self, individual):
        """
        Evaluate a single individual using PINN
        
        Args:
            individual: dict with keys 'cp1', 'cp2', 'cp3', 'Re', 'Sc'
            
        Returns:
            efficiency: fitness value (higher is better)
        """
        cp1, cp2, cp3, Re, Sc = individual['cp1'], individual['cp2'], individual['cp3'], individual['Re'], individual['Sc']
        
        try:
            # Create evaluation points (similar to RL approach)
            # Inlet points for pressure evaluation
            xi = np.linspace(0.0, 1.0, NUM_EVAL_POINTS)
            yi = np.array([1.5])  # Inlet y-position
            p1_vals = np.array([cp1])
            p2_vals = np.array([cp2])
            p3_vals = np.array([cp3])
            re_vals = np.array([Re])
            sc_vals = np.array([Sc])
            
            # Create meshgrid for inlet
            X_i, Y_i, P1_i, P2_i, P3_i, RE_i, SC_i = np.meshgrid(xi, yi, p1_vals, p2_vals, p3_vals, re_vals, sc_vals)
            inlet_points = np.stack([X_i.ravel(), Y_i.ravel(), P1_i.ravel(), P2_i.ravel(), 
                                   P3_i.ravel(), RE_i.ravel(), SC_i.ravel()], axis=1)
            inlet_points = torch.tensor(inlet_points, dtype=torch.float32).to(self.device)
            
            # Outlet points for concentration evaluation
            xo = np.array([7.0])  # Outlet x-position
            yo = np.linspace(0.5, 1.5, NUM_EVAL_POINTS)
            p1_vals = np.array([cp1])
            p2_vals = np.array([cp2])
            p3_vals = np.array([cp3])
            re_vals = np.array([Re])
            sc_vals = np.array([Sc])
            
            # Create meshgrid for outlet
            X_o, Y_o, P1_o, P2_o, P3_o, RE_o, SC_o = np.meshgrid(xo, yo, p1_vals, p2_vals, 
                                                                 p3_vals, re_vals, sc_vals)
            outlet_points = np.stack([X_o.ravel(), Y_o.ravel(), P1_o.ravel(), P2_o.ravel(), 
                                    P3_o.ravel(), RE_o.ravel(), SC_o.ravel()], axis=1)
            outlet_points = torch.tensor(outlet_points, dtype=torch.float32).to(self.device)
            
            # PINN evaluation (placeholder - replace with actual PINN)
            with torch.no_grad():
                # Replace with your actual PINN predictions
                # concentration = pinn.predict(outlet_points)[6]  # Concentration
                # pressure = pinn.predict(inlet_points)[2]       # Pressure
                
                # Placeholder values for demonstration
                concentration = torch.randn(outlet_points.shape[0], 1)
                pressure = torch.randn(inlet_points.shape[0], 1)
                
                concentration = concentration.cpu().numpy()
                pressure = pressure.cpu().numpy()
            
            # Calculate efficiency metric
            mixing_index = self.calculate_mixing_index(concentration)
            pressure_drop = np.mean(np.abs(pressure))
            
            # Avoid division by zero
            if pressure_drop < 1e-10:
                efficiency = 0.0
            else:
                efficiency = mixing_index / (pressure_drop ** (1/3))
            
            # Clip efficiency to reasonable range
            efficiency = np.clip(efficiency, 0.0, 1.0)
            
            return efficiency
            
        except Exception as e:
            print(f"Error evaluating individual: {e}")
            return 0.0  # Return minimum fitness for invalid individuals