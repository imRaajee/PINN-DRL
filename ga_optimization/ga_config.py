"""
Configuration for Genetic Algorithm optimization
"""

import numpy as np

# Genetic Algorithm parameters
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 5

# Parameter bounds
PARAMETER_BOUNDS = {
    'cp1': [0.0, 1.0],    # Control point 1
    'cp2': [0.0, 1.0],    # Control point 2  
    'cp3': [0.0, 1.0],    # Control point 3
    'Re': [5.0, 80.0],    # Reynolds number
    'Sc': [0.1, 10.0]     # Schmidt number
}

# Evaluation parameters
NUM_EVAL_POINTS = 100  # Number of points for PINN evaluation