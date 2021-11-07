from typing import Any, Dict, List
from experiments.logger import Logger
from games.level import Level
from novelty_neat.fitness.fitness import NeatFitnessFunction


class DirectFitness:
    """
        This is a fitness function that uses a NeatFitnessFunction to simply calculate fitnesses for individual levels.
    """
    def __init__(self, neat_fitness_function: NeatFitnessFunction):
        self.neat_fitness_function = neat_fitness_function
    
    def calc_fitness(self, levels: List[Level], logger: Logger = None) -> List[float]:
        levels_stacked = [[l] for l in levels]
        if logger is not None:
            self.neat_fitness_function.logger = logger
        self.neat_fitness_function.steps += 1
        k = self.neat_fitness_function.calc_fitness(None, levels_stacked)
        return k
    
    def params(self) -> Dict[str, Any]:
         return {
             'name': "DirectFitness",
             "sub": self.neat_fitness_function.params()
         }
    
    def reset(self):
        self.neat_fitness_function.reset()
        