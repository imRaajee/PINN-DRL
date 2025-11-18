"""
Main script to run the PINN collocation points generation
"""

from collocation_points import generate_collocation_points, plot_collocation_points

if __name__ == "__main__":
    # Generate collocation points
    col = generate_collocation_points()
    print(f"Shape of collocation points: {np.shape(col)}")
    
    # Visualize the points
    plot_collocation_points(col)