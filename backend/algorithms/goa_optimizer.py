import numpy as np
from typing import Callable, List, Dict, Tuple
import logging

class GrasshopperOptimizer:
    """
    Grasshopper Optimization Algorithm implementation for watermark scaling factor optimization.
    """
    
    def __init__(self, 
                n_grasshoppers: int = 30,
                max_iterations: int = 100,
                c_min: float = 0.00001,
                c_max: float = 1.0,
                f_min: float = 0.01,
                f_max: float = 1.0):
        """
        Initialize GOA optimizer
        
        Args:
            n_grasshoppers: Number of search agents
            max_iterations: Maximum iterations
            c_min: Minimum value of c parameter
            c_max: Maximum value of c parameter
            f_min: Lower bound of search space
            f_max: Upper bound of search space
        """
        self.n_grasshoppers = n_grasshoppers
        self.max_iterations = max_iterations
        self.c_min = c_min
        self.c_max = c_max
        self.f_min = f_min
        self.f_max = f_max
        
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_curve = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def optimize(self, objective_function: Callable, dims: int = 1) -> Dict:
        """
        Execute GOA optimization
        
        Args:
            objective_function: Function to minimize
            dims: Number of dimensions (usually 1 for scaling factor)
            
        Returns:
            Dict containing:
                - best_solution: Optimal scaling factor
                - best_fitness: Best fitness value
                - convergence: Convergence history
        """
        # Initialize population
        population = np.random.uniform(
            self.f_min, 
            self.f_max, 
            (self.n_grasshoppers, dims)
        )
        
        # Reset best solution
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Update c parameter
            c = self.c_max - iteration * ((self.c_max - self.c_min) / self.max_iterations)
            
            # Evaluate fitness for all grasshoppers
            # Handle both scalar and array inputs for objective function
            fitness_values = []
            for pos in population:
                try:
                    alpha_val = pos[0] if isinstance(pos, np.ndarray) and len(pos) > 0 else pos
                    fitness = objective_function(alpha_val)
                    # Handle infinite or NaN values
                    if not np.isfinite(fitness):
                        fitness = 1e10
                    fitness_values.append(fitness)
                except Exception as e:
                    self.logger.warning(f"Fitness evaluation failed: {str(e)}")
                    fitness_values.append(1e10)
            
            fitness_values = np.array(fitness_values)
            
            # Update best solution
            current_best_idx = np.argmin(fitness_values)
            if fitness_values[current_best_idx] < self.best_fitness:
                self.best_fitness = fitness_values[current_best_idx]
                self.best_solution = population[current_best_idx].copy()
            
            # Initialize best_solution if not set (first iteration or all values were bad)
            if self.best_solution is None:
                self.best_solution = population[current_best_idx].copy()
                self.best_fitness = fitness_values[current_best_idx]
            
            self.convergence_curve.append(self.best_fitness)
            
            # Update grasshopper positions
            for i in range(self.n_grasshoppers):
                social_interaction = np.zeros(dims)
                
                for j in range(self.n_grasshoppers):
                    if i != j:
                        distance = np.abs(population[j] - population[i])
                        s_function = (self.f_max - self.f_min) * distance / 2
                        direction = (population[j] - population[i]) / (distance + 1e-10)
                        social_interaction += s_function * direction
                
                # Use best_solution for position update
                next_position = c * social_interaction + self.best_solution
                population[i] = np.clip(next_position, self.f_min, self.f_max)
            
            self.logger.info(f"Iteration {iteration + 1}/{self.max_iterations}, "
                           f"Best fitness: {self.best_fitness:.6f}")
        
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'convergence': self.convergence_curve
        }