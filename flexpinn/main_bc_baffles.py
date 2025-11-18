"""
Main script to run the PINN data generation
"""

from collocation_points import generate_collocation_points, plot_collocation_points
from boundary_conditions import generate_wall_boundaries, generate_inlet_outlet, generate_sections
from baffles import generate_baffles_and_normals, plot_baffles

if __name__ == "__main__":
    # Generate collocation points
    col = generate_collocation_points()
    print(f"Collocation points shape: {np.shape(col)}")
    
    # Generate boundary conditions
    wall_up, wall_dn, wall_left = generate_wall_boundaries()
    inout = generate_inlet_outlet()
    sections = generate_sections()
    
    print(f"Upper wall points: {np.shape(wall_up)}")
    print(f"Lower wall points: {np.shape(wall_dn)}")
    print(f"Left wall points: {np.shape(wall_left)}")
    print(f"Inlet/Outlet points shape: {np.shape(inout)}")
    print(f"Section points shape: {np.shape(sections)}")
    
    # Generate baffles and normals
    Fins, Normals = generate_baffles_and_normals()
    print(f"Baffle points shape: {np.shape(Fins)}")
    print(f"Normal vectors shape: {np.shape(Normals)}")
    
    # Visualize the points
    plot_collocation_points(col)
    plot_baffles(Fins)