"""
Aggregate and prepare all data for PINN training
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from config import device

def aggregate_data(wall_dn, wall_up, wall_left, Fins, inlet_1, inlet_2, outlet, secs, col):
    """Aggregate all boundary and collocation data"""
    # Combine wall boundaries
    walls = np.concatenate((wall_dn, wall_up, wall_left, Fins), axis=0)
    
    # Set wall velocity boundary conditions (no-slip)
    wall_uv = np.full_like(walls[:, 0:2], 0)
    
    # Calculate inlet velocity profiles (parabolic)
    inlet_v1 = 4 * inlet_1[:, 0:1] * (d - inlet_1[:, 0:1]) / (d ** 2)
    inlet_v2 = -4 * inlet_2[:, 0:1] * (d - inlet_2[:, 0:1]) / (d ** 2)
    
    inlet_u1 = np.full_like(inlet_1[:, 0:1], 0.0)
    inlet_uv1 = np.concatenate([inlet_u1, inlet_v1], axis=1)
    inlet_u2 = np.full_like(inlet_2[:, 0:1], 0.0)
    inlet_uv2 = np.concatenate([inlet_u2, inlet_v2], axis=1)
    
    # Jacobian boundary points
    wall_Jx = wall_left
    wall_Jy = np.concatenate((wall_dn, wall_up), axis=0)
    
    # Combine all boundary conditions
    bc = np.concatenate([inlet_1, inlet_2, walls], axis=0)
    bc_uv = np.concatenate([inlet_uv1, inlet_uv2, wall_uv], axis=0)
    
    return walls, wall_uv, inlet_uv1, inlet_uv2, wall_Jx, wall_Jy, bc, bc_uv

def convert_to_tensors(inout, bc, bc_uv, col, wall_Jx, wall_Jy, Fins, All_Normals, secs):
    """Convert all numpy arrays to PyTorch tensors"""
    inout_tensor = torch.tensor(inout, dtype=torch.float32).to(device)
    bc_tensor = torch.tensor(bc, dtype=torch.float32).to(device)
    bc_uv_tensor = torch.tensor(bc_uv, dtype=torch.float32).to(device)
    col_tensor = torch.tensor(col, dtype=torch.float32).to(device)
    wall_Jx_tensor = torch.tensor(wall_Jx, dtype=torch.float32).to(device)
    wall_Jy_tensor = torch.tensor(wall_Jy, dtype=torch.float32).to(device)
    Fins_tensor = torch.tensor(Fins, dtype=torch.float32).to(device)
    Normals_tensor = torch.tensor(All_Normals, dtype=torch.float32).to(device)
    secs_tensor = torch.tensor(secs, dtype=torch.float32).to(device)
    
    return (inout_tensor, bc_tensor, bc_uv_tensor, col_tensor, 
            wall_Jx_tensor, wall_Jy_tensor, Fins_tensor, Normals_tensor, secs_tensor)

def print_data_shapes(col, walls, inout, bc_uv, secs, Fins):
    """Print shapes of all data arrays"""
    print('Collocation points: ', np.shape(col))
    print('Wall points: ', np.shape(walls))
    print('Inlet/Outlet points: ', np.shape(inout))
    print('BC velocity points: ', np.shape(bc_uv))
    print('Section points: ', np.shape(secs))
    print('Fins points: ', np.shape(Fins))

def plot_geometry(col, bc, Fins, wall_Jx, wall_Jy, inlet_1, inlet_2, outlet, secs):
    """Plot the complete geometry with all point types"""
    fig, ax = plt.subplots(dpi=100)
    ax.set_aspect("equal")
    
    # Plot different point types with different colors
    plt.scatter(col[:, 0], col[:, 1], color='pink', marker='.', alpha=0.5, label='Collocation')
    plt.scatter(bc[:, 0], bc[:, 1], color='blue', marker='.', alpha=0.5, label='BC')
    plt.scatter(Fins[:, 0], Fins[:, 1], color='red', marker='.', alpha=0.5, label='Fins')
    plt.scatter(wall_Jx[:, 0], wall_Jx[:, 1], color='purple', marker='.', alpha=0.5, label='Wall Jx')
    plt.scatter(wall_Jy[:, 0], wall_Jy[:, 1], color='purple', marker='.', alpha=0.5)
    plt.scatter(inlet_1[:, 0], inlet_1[:, 1], color='black', marker='.', alpha=0.5, label='Inlet/Outlet')
    plt.scatter(inlet_2[:, 0], inlet_2[:, 1], color='black', marker='.', alpha=0.5)
    plt.scatter(outlet[:, 0], outlet[:, 1], color='black', marker='.', alpha=0.5)
    plt.scatter(secs[:, :, 0], secs[:, :, 1], color='yellow', marker='.', alpha=0.5, label='Sections')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Geometry Overview')
    plt.legend()
    plt.show()

def plot_boundary_conditions(bc, bc_uv):
    """Plot boundary condition velocities"""
    fig, ax = plt.subplots()
    plt.scatter(bc[:, 0], bc_uv[:, 1], marker='.', alpha=0.1, color='blue')
    plt.xlabel('x position')
    plt.ylabel('y velocity')
    plt.title('Boundary Condition Velocities')
    plt.show()