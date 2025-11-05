"""
Main script to run FlexPINN training
"""

import time
from collocation_points import generate_collocation_points
from boundary_conditions import generate_wall_boundaries, generate_inlet_outlet, generate_sections
from baffles import generate_baffles_and_normals
from data_aggregation import aggregate_data, convert_to_tensors, print_data_shapes, plot_geometry
from flex_pinn import PINN

if __name__ == "__main__":
    # Generate all data
    print("Generating geometry and training data...")
    col = generate_collocation_points()
    wall_up, wall_dn, wall_left = generate_wall_boundaries()
    inout = generate_inlet_outlet()
    sections = generate_sections()
    Fins, All_Normals = generate_baffles_and_normals()
    
    # Extract individual components
    inlet_1 = inout[:, :, 0]
    inlet_2 = inout[:, :, 1] 
    outlet = inout[:, :, 2]
    
    # Aggregate data
    walls, wall_uv, inlet_uv1, inlet_uv2, wall_Jx, wall_Jy, bc, bc_uv = aggregate_data(
        wall_dn, wall_up, wall_left, Fins, inlet_1, inlet_2, outlet, sections, col
    )
    
    # Print data summary
    print_data_shapes(col, walls, inout, bc_uv, sections, Fins)
    
    # Convert to tensors
    tensors = convert_to_tensors(inout, bc, bc_uv, col, wall_Jx, wall_Jy, Fins, All_Normals, sections)
    (inout_tensor, bc_tensor, bc_uv_tensor, col_tensor, 
     wall_Jx_tensor, wall_Jy_tensor, Fins_tensor, Normals_tensor, secs_tensor) = tensors
    
    print("Data preparation complete. Starting training...")
    
    # Initialize and train PINN
    pinn = PINN()
    pinn.train(bc_tensor, wall_Jx_tensor, wall_Jy_tensor, Fins_tensor, 
               Normals_tensor, inout_tensor, col_tensor, secs_tensor)
    
    print("Training completed!")