"""
Main script to run the PINN data generation and visualization
"""

import numpy as np
from collocation_points import generate_collocation_points
from boundary_conditions import generate_wall_boundaries, generate_inlet_outlet, generate_sections
from baffles import generate_baffles_and_normals
from data_aggregation import (aggregate_data, convert_to_tensors, 
                             print_data_shapes, plot_geometry, plot_boundary_conditions)

if __name__ == "__main__":
    # Generate all data components
    col = generate_collocation_points()
    wall_up, wall_dn, wall_left = generate_wall_boundaries()
    inout = generate_inlet_outlet()
    sections = generate_sections()
    Fins, All_Normals = generate_baffles_and_normals()
    
    # Extract individual components from inout
    inlet_1 = inout[:, :, 0]
    inlet_2 = inout[:, :, 1] 
    outlet = inout[:, :, 2]
    
    # Aggregate all data
    walls, wall_uv, inlet_uv1, inlet_uv2, wall_Jx, wall_Jy, bc, bc_uv = aggregate_data(
        wall_dn, wall_up, wall_left, Fins, inlet_1, inlet_2, outlet, sections, col
    )
    
    # Print data shapes
    print_data_shapes(col, walls, inout, bc_uv, sections, Fins)
    
    # Convert to PyTorch tensors
    tensors = convert_to_tensors(inout, bc, bc_uv, col, wall_Jx, wall_Jy, Fins, All_Normals, sections)
    (inout_tensor, bc_tensor, bc_uv_tensor, col_tensor, 
     wall_Jx_tensor, wall_Jy_tensor, Fins_tensor, Normals_tensor, secs_tensor) = tensors
    
    print("All data converted to PyTorch tensors on device:", device)
    
    # Visualize results
    plot_geometry(col, bc, Fins, wall_Jx, wall_Jy, inlet_1, inlet_2, outlet, sections)
    plot_boundary_conditions(bc, bc_uv)