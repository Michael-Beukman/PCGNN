import os
import pickle
from common.methods.pcg_method import PCGMethod
from typing import Any, Dict, List, Callable, Tuple, Union
from common.types import Verbosity
from common.utils import get_date
from games.game import Game
from games.level import Level
from experiments.logger import Logger
import neat
import numpy as np
from novelty_neat.fitness.fitness import NeatFitnessFunction
from novelty_neat.generation import NeatLevelGenerator
from novelty_neat.types import LevelNeuralNet


# This function returns a single level given some input and a neural network.

# This function takes in a list of networks and returns a list of floats representing their fitnesses.
# Since it's a callable, it can be a class that stores some state.


class NoveltyNeatPCG(PCGMethod):
    """This is the method that uses NEAT to evolve a neural network to generate levels, 
        and uses novelty search as the fitness function to ensure individuals are valid.

    """
    def __init__(self, game: Game, init_level: Level, level_generator: NeatLevelGenerator, 
                    fitness_calculator: NeatFitnessFunction, neat_config: neat.Config,
                    num_generations: int=10, num_random_vars=2
                    ) -> None:
        """Relatively general constructor, where all interesting behaviour can be provided using different callables.

        Args:
            game (Game): The game that levels should be generated for
            init_level (Level): The initial level to use as a starting point
            level_generator (NeatLevelGenerator): This should take in some input, and a network and return a Level.
            fitness_calculator (NeatFitnessFunction): This should take in a list of networks and return a list of fitnesses.
            neat_config (neat.Config): The configuration used for the NEAT algorithm.
            
            num_generations (int): How many generations to train for.
            num_random_vars (int): How many random variables should we use as input to the generation process.
        """
        super().__init__(game, init_level)
        self.level_generator = level_generator
        self.fitness_calculator = fitness_calculator
        self.neat_config = neat_config
        self.pop = neat.Population(self.neat_config)
        self.best_agent: Union[neat.DefaultGenome, None] = None
        
        self.num_generations = num_generations
        self.num_random_vars = num_random_vars
    
    def train(self, logger: Logger) -> List[Dict[str, Any]]:
        """The training procedure to follow. We basically just call `run` on self.population 
             using the functions provided.

        Args:
            logger (Logger): 

        Returns:
            List[Dict[str, Any]]: 
        """
        # This is the training procedure.
        steps = 0
        def fitness(genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config):
            nonlocal steps
            nets = []
            for genome_id, genome in genomes:
                nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
            all_fitnesses = self.fitness_calculator(nets)
            for fit, (_, genome) in zip(all_fitnesses, genomes):
                genome.fitness = fit
            # Log some info.
            logger.log({
                'mean_fitness': np.mean(all_fitnesses),
                'max_fitness': np.max(all_fitnesses),
                'min_fitness': np.min(all_fitnesses),
            },step=steps)
            steps += 1
            if logger.verbose == Verbosity.PROGRESS:
                print(f"\r{steps} / {self.num_generations}", end='')

        self.fitness_calculator.logger = logger
        self.best_agent = self.pop.run(fitness_function=fitness, n=self.num_generations)
        if logger.verbose == Verbosity.PROGRESS:
            print("")
        return [{'final_agent': self.best_agent, 'population': self.pop}]
    
    def generate_level(self) -> Level:
        """Simple generates a level by calling level_generator with self.best_agent.

        Returns:
            Level: 
        """
        assert self.best_agent is not None, "self.best_agent should not be None. Run train first"
        return self.level_generator(neat.nn.FeedForwardNetwork.create(self.best_agent, self.neat_config))
    
    @classmethod
    def name(cls):
        """
            Returns a name of this class
        """
        return str(cls.__name__)

    def save_best_individual(self) -> str:
        """ Saves the best individual to a file 'results/scratch/neat_novelty/{self.name()}/{get_date()}/best.p'
        """
        folder = f"results/scratch/neat_novelty/{self.name()}/{get_date()}"
        file = os.path.join(folder, 'best.p')
        os.makedirs(folder, exist_ok=True)
        with open(file, 'wb+') as f:
            pickle.dump({'best_individual': self.best_agent, 'config': self.neat_config}, f)
        return file