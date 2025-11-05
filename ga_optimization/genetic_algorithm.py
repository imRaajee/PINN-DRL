"""
Genetic Algorithm implementation for PINN optimization
"""

import numpy as np
import random
from typing import List, Dict, Tuple
from ga_config import *
from evaluator import PINNEvaluator

class GeneticAlgorithm:
    """Genetic Algorithm for optimizing PINN parameters"""
    
    def __init__(self):
        self.evaluator = PINNEvaluator()
        self.population = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.fitness_history = []
        self.diversity_history = []
        
    def initialize_population(self) -> List[Dict]:
        """Initialize random population within parameter bounds"""
        population = []
        for _ in range(POPULATION_SIZE):
            individual = {
                'cp1': np.random.uniform(*PARAMETER_BOUNDS['cp1']),
                'cp2': np.random.uniform(*PARAMETER_BOUNDS['cp2']),
                'cp3': np.random.uniform(*PARAMETER_BOUNDS['cp3']),
                'Re': np.random.uniform(*PARAMETER_BOUNDS['Re']),
                'Sc': np.random.uniform(*PARAMETER_BOUNDS['Sc'])
            }
            population.append(individual)
        return population
    
    def evaluate_population(self, population: List[Dict]) -> List[float]:
        """Evaluate fitness for entire population"""
        fitness_scores = []
        for individual in population:
            fitness = self.evaluator.evaluate_individual(individual)
            fitness_scores.append(fitness)
        return fitness_scores
    
    def select_parents(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Select parents using tournament selection"""
        parents = []
        
        # Always include elite individuals
        elite_indices = np.argsort(fitness_scores)[-ELITISM_COUNT:]
        for idx in elite_indices:
            parents.append(population[idx])
        
        # Tournament selection for remaining parents
        while len(parents) < POPULATION_SIZE:
            # Randomly select tournament participants
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select winner (highest fitness)
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        
        return parents
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Perform crossover between two parents"""
        if random.random() > CROSSOVER_RATE:
            return parent1, parent2
        
        child1, child2 = {}, {}
        
        # Blend crossover for continuous parameters
        alpha = random.random()
        
        for param in PARAMETER_BOUNDS.keys():
            # Linear interpolation between parent values
            child1[param] = alpha * parent1[param] + (1 - alpha) * parent2[param]
            child2[param] = (1 - alpha) * parent1[param] + alpha * parent2[param]
            
            # Ensure bounds are respected
            child1[param] = np.clip(child1[param], *PARAMETER_BOUNDS[param])
            child2[param] = np.clip(child2[param], *PARAMETER_BOUNDS[param])
        
        return child1, child2
    
    def mutate(self, individual: Dict) -> Dict:
        """Apply mutation to an individual"""
        mutated = individual.copy()
        
        for param in PARAMETER_BOUNDS.keys():
            if random.random() < MUTATION_RATE:
                # Gaussian mutation
                current_value = individual[param]
                lower_bound, upper_bound = PARAMETER_BOUNDS[param]
                range_size = upper_bound - lower_bound
                
                # Mutate with normal distribution (5% of parameter range)
                mutation_strength = 0.05 * range_size
                new_value = current_value + random.gauss(0, mutation_strength)
                
                # Clip to bounds
                mutated[param] = np.clip(new_value, lower_bound, upper_bound)
        
        return mutated
    
    def calculate_population_diversity(self, population: List[Dict]) -> float:
        """Calculate diversity of population (average pairwise distance)"""
        if len(population) <= 1:
            return 0.0
        
        # Convert population to numpy array
        pop_array = np.array([[ind['cp1'], ind['cp2'], ind['cp3'], ind['Re'], ind['Sc']] 
                             for ind in population])
        
        # Calculate pairwise Euclidean distances
        total_distance = 0
        count = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.linalg.norm(pop_array[i] - pop_array[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def run(self) -> Dict:
        """Run the genetic algorithm optimization"""
        print("Initializing Genetic Algorithm...")
        
        # Initialize population
        self.population = self.initialize_population()
        
        print(f"Running GA for {GENERATIONS} generations...")
        
        for generation in range(GENERATIONS):
            # Evaluate population
            fitness_scores = self.evaluate_population(self.population)
            
            # Update best individual
            best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[best_idx]
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = self.population[best_idx].copy()
            
            # Store history
            self.fitness_history.append({
                'generation': generation,
                'best': np.max(fitness_scores),
                'average': np.mean(fitness_scores),
                'worst': np.min(fitness_scores)
            })
            
            # Calculate diversity
            diversity = self.calculate_population_diversity(self.population)
            self.diversity_history.append(diversity)
            
            # Print progress
            if generation % 10 == 0:
                print(f"Generation {generation:3d}: "
                      f"Best={self.fitness_history[-1]['best']:.4f}, "
                      f"Avg={self.fitness_history[-1]['average']:.4f}, "
                      f"Diversity={diversity:.4f}")
            
            # Selection
            parents = self.select_parents(self.population, fitness_scores)
            
            # Create new generation
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[-ELITISM_COUNT:]
            for idx in elite_indices:
                new_population.append(self.population[idx])
            
            # Crossover and mutation
            while len(new_population) < POPULATION_SIZE:
                # Select two parents
                parent1, parent2 = random.sample(parents, 2)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Ensure population size is correct
            self.population = new_population[:POPULATION_SIZE]
        
        print("Genetic Algorithm completed!")
        return self.best_individual