"""
Visualization utilities for Genetic Algorithm results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_ga_results(ga_instance, save_path="ga_results.png"):
    """Create comprehensive visualization of GA results"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Extract data
    generations = [fh['generation'] for fh in ga_instance.fitness_history]
    best_fitness = [fh['best'] for fh in ga_instance.fitness_history]
    avg_fitness = [fh['average'] for fh in ga_instance.fitness_history]
    worst_fitness = [fh['worst'] for fh in ga_instance.fitness_history]
    diversity = ga_instance.diversity_history
    
    # 1. Fitness progression
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(generations, best_fitness, 'g-', linewidth=2, label='Best Fitness')
    ax1.plot(generations, avg_fitness, 'b-', linewidth=2, label='Average Fitness')
    ax1.plot(generations, worst_fitness, 'r-', linewidth=1, label='Worst Fitness', alpha=0.7)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Diversity progression
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(generations, diversity, 'purple', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Population Diversity')
    ax2.set_title('Population Diversity Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Final population distribution
    ax3 = plt.subplot(2, 3, 3)
    final_population = ga_instance.population
    parameters = ['cp1', 'cp2', 'cp3', 'Re', 'Sc']
    parameter_names = ['Control Point 1', 'Control Point 2', 'Control Point 3', 'Reynolds Number', 'Schmidt Number']
    
    data = []
    for param, name in zip(parameters, parameter_names):
        values = [ind[param] for ind in final_population]
        data.append(values)
    
    box_plot = ax3.boxplot(data, labels=parameter_names, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'wheat', 'plum']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('Final Population Parameter Distribution')
    plt.xticks(rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Best solution parameters
    ax4 = plt.subplot(2, 3, 4)
    best_ind = ga_instance.best_individual
    param_values = [best_ind[param] for param in parameters]
    
    bars = ax4.bar(parameter_names, param_values, color=colors, alpha=0.7)
    ax4.set_ylabel('Parameter Value')
    ax4.set_title('Best Solution Parameters')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, param_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    
    # 5. Parameter correlations
    ax5 = plt.subplot(2, 3, 5)
    pop_array = np.array([[ind['cp1'], ind['cp2'], ind['cp3'], ind['Re'], ind['Sc']] 
                         for ind in final_population])
    correlation_matrix = np.corrcoef(pop_array.T)
    
    im = ax5.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax5.set_xticks(range(len(parameters)))
    ax5.set_yticks(range(len(parameters)))
    ax5.set_xticklabels(parameters, rotation=45)
    ax5.set_yticklabels(parameters)
    ax5.set_title('Parameter Correlation Matrix')
    
    # Add correlation values
    for i in range(len(parameters)):
        for j in range(len(parameters)):
            ax5.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax5)
    
    # 6. Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    Genetic Algorithm Results:
    --------------------------
    Total Generations: {GENERATIONS}
    Population Size: {POPULATION_SIZE}
    Best Fitness: {ga_instance.best_fitness:.6f}
    
    Best Solution:
    CP1: {best_ind['cp1']:.4f}
    CP2: {best_ind['cp2']:.4f} 
    CP3: {best_ind['cp3']:.4f}
    Re:  {best_ind['Re']:.2f}
    Sc:  {best_ind['Sc']:.2f}
    
    Final Generation:
    Best Fitness: {best_fitness[-1]:.6f}
    Avg Fitness: {avg_fitness[-1]:.6f}
    Diversity: {diversity[-1]:.4f}
    """
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def save_results_to_csv(ga_instance, filename="ga_results.csv"):
    """Save GA results to CSV file"""
    # Save fitness history
    fitness_df = pd.DataFrame(ga_instance.fitness_history)
    fitness_df['diversity'] = ga_instance.diversity_history
    fitness_df.to_csv(f"fitness_{filename}", index=False)
    
    # Save final population
    pop_df = pd.DataFrame(ga_instance.population)
    pop_df.to_csv(f"population_{filename}", index=False)
    
    # Save best solution
    best_df = pd.DataFrame([ga_instance.best_individual])
    best_df['fitness'] = ga_instance.best_fitness
    best_df.to_csv(f"best_{filename}", index=False)
    
    print(f"Results saved to CSV files")