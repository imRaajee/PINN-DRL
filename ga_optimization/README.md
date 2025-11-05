# Genetic Algorithm Optimization

Genetic Algorithm implementation for optimizing mixer geometry parameters using trained PINN models as evaluators.

## Files

- **`ga_config.py`** - GA parameters and parameter bounds
- **`genetic_algorithm.py`** - Core GA with tournament selection and elitism
- **`evaluator.py`** - PINN-based fitness evaluation
- **`visualization.py`** - Results analysis and plotting
- **`run_ga.py`** - Main optimization script

## Features
- Tournament selection with elitism
- Blend crossover and Gaussian mutation
- Population diversity tracking
- Comprehensive visualization of results
