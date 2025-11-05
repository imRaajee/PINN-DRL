"""
Main script to run Genetic Algorithm optimization
"""

import time
from genetic_algorithm import GeneticAlgorithm
from visualization import plot_ga_results, save_results_to_csv

def main():
    """Run Genetic Algorithm optimization for PINN parameters"""
    print("Starting Genetic Algorithm Optimization for PINN")
    print("=" * 50)
    
    start_time = time.time()
    
    # Initialize and run GA
    ga = GeneticAlgorithm()
    best_solution = ga.run()
    
    end_time = time.time()
    
    # Display results
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETED")
    print("=" * 50)
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Best fitness: {ga.best_fitness:.6f}")
    print("\nBest solution found:")
    print(f"  Control Point 1: {best_solution['cp1']:.4f}")
    print(f"  Control Point 2: {best_solution['cp2']:.4f}")
    print(f"  Control Point 3: {best_solution['cp3']:.4f}")
    print(f"  Reynolds Number: {best_solution['Re']:.2f}")
    print(f"  Schmidt Number:  {best_solution['Sc']:.2f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_ga_results(ga, "ga_optimization_results.png")
    
    # Save results to CSV
    save_results_to_csv(ga, "ga_optimization_results.csv")
    
    print("\nAll results saved successfully!")

if __name__ == "__main__":
    main()