import copy
from math import ceil
from typing import Callable, List, Tuple
from matplotlib import pyplot as plt

import numpy as np
from baselines.ga.direct_ga_fitness import DirectFitness
from baselines.ga.general_ga_pcg import _GeneralGAIndividual
from common.methods.pcg_method import PCGMethod
from games.game import Game
from games.level import Level
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from gym_pcgrl.envs.helper import calc_longest_path, get_tile_locations
from skimage import morphology as morph
import scipy
from metrics.horn.leniency import LeniencyMetric
from novelty_neat.fitness.entropy import EntropyFitness
from novelty_neat.fitness.fitness import CombinedFitness
from novelty_neat.maze.neat_maze_fitness import SolvabilityFitness
from novelty_neat.novelty.distance_functions.distance import visual_diversity
from novelty_neat.novelty.novelty_metric import NoveltyMetric

class SingleElementGAIndividual:
    def __init__(self, level: Level, min: int, max: int, init: int = 0) -> None:
        """A single individual that handles a single aspect of a level. 

        Args:
            level (Level): NOT USED
            min (int): The smallest value that these tiles can take on
            max (int): The largest value (inclusive) that these tiles can take on
            init (int, optional): The initial value. Defaults to 0.
        """
        self.level = level
        self.genome = np.zeros(level.width, dtype=np.int32) + init
        self.min = min
        self.max = max
    
    def crossover(self, other: "SingleElementGAIndividual") -> "SingleElementGAIndividual":
        k = np.random.randint(0, len(self.genome) - 1)
        new_genome = np.zeros_like(self.genome)
        new_genome2 = np.zeros_like(self.genome)

        new_genome[:k] += self.genome[:k]
        new_genome[k:] += other.genome[k:]

        new_genome2[:k] += other.genome[:k]
        new_genome2[k:] += self.genome[k:]

        new_agent = SingleElementGAIndividual(self.level, min=self.min, max=self.max)
        new_agent.genome = new_genome

        new_agent2 = SingleElementGAIndividual(self.level, min=self.min, max=self.max)
        new_agent2.genome = new_genome2

        return new_agent, new_agent2

    def mutate(self, prob):
        indices = np.random.rand(len(self.genome)) < prob
        random_ints = np.random.randint(self.min, self.max + 1, size=indices.sum())
        self.genome[indices] = random_ints

class SingleElementFitnessFunction:
    pass
    def calc_fitness(self, individuals: List[SingleElementGAIndividual]) -> List[float]:
        raise NotImplementedError("")

    def reset(self):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class MultiPopGAPCG(PCGMethod):
    """
        This is a method that attempt to replicate
            Ferreira, L., Pereira, L., & Toledo, C. (2014, July). A multi-population genetic algorithm for procedural generation of levels for platform games. In Proceedings of the Companion Publication of the 2014 Annual Conference on Genetic and Evolutionary Computation (pp. 45-46).
        Specifically using multiple populations, and at the end we combine them into one level
    """

    def __init__(self, game: Game, init_level: Level,
                 individual_funcs: List[Callable[[Level], SingleElementGAIndividual]],
                 fitness_functions: List[SingleElementFitnessFunction], 
                 population_size: int = 50,
                 number_of_generations: int = 100) -> None:
        super().__init__(game, init_level)
        self.population_size = population_size
        self.individual_funcs = individual_funcs
        self.level = init_level
        self.number_of_generations = number_of_generations
        self.fitness_functions = fitness_functions
        self.reset()
        assert len(self.fitness_functions) == len(self.individual_funcs) == len(self.populations)

    def reset(self):
        self.populations: List[List[SingleElementGAIndividual]] = \
        [
            [individual_func(copy.deepcopy(self.level)) for i in range(self.population_size)]
            for individual_func in self.individual_funcs
        ]
        self.gen_count = 0
        self.best_agent = None
        for f in self.fitness_functions:
            f.reset()

    def one_gen(self):
        """Performs one generation, evaluates and breeds.
        """
        for index, (pop, fit) in enumerate(zip(self.populations, self.fitness_functions)):
            probs = self.evaluate(pop, fit)
            self.populations[index] = self.breed(pop, probs)
        self.gen_count += 1

    def breed(self, pop: List[SingleElementGAIndividual], probs: List[float]):
        """Breeds the current population to form the next one. 
           The individual with the best score is kept unchanged, and the others are from crossover with parents.

        Args:
            probs (List[float]): With what probability should each parent be chosen.
        """
        best_agent = np.argmax(probs)
        best_agent = pop[best_agent]
        new_pop = [best_agent]

        while len(new_pop) < self.population_size:
            a1: SingleElementGAIndividual = np.random.choice(pop, p=probs)
            a2: SingleElementGAIndividual = np.random.choice(pop, p=probs)
            c = 0
            while a2 == a1 and c < 5:
                c += 1
                a2 = np.random.choice(pop, p=probs)

            if (a1 == a2):
                print(" # ", end='')
            child1, child2 = a1.crossover(a2)
            child1.mutate(0.2)
            child2.mutate(0.2)
            new_pop.append(child1)
            if len(new_pop) < self.population_size:
                new_pop.append(child2)
        return new_pop

    def evaluate(self, pop: List[SingleElementGAIndividual], fitness: SingleElementFitnessFunction) -> List[float]:
        """This simply goes through the population and calculates their fitnesses.
           It normalises these fitnesses, so that sum(evaluate()) == 1.

        Returns:
            List[float]: Probability of each parent to be chosen for breeding.
        """
        total_fit = 0
        max_fit = -1
        fitnesses = fitness.calc_fitness(pop)

        for a, f in zip(pop, fitnesses):
            a.fitness = f
            max_fit = max(max_fit, a.fitness)
            total_fit += a.fitness

        self.best_fit = max_fit
        if total_fit != 0:
            probs = [
                a.fitness / total_fit for a in pop
            ]
        else:
            probs = [a.fitness for a in pop]
            probs[0] = 1
        return probs

    def generate_level(self) -> Level:
        self.reset()
        for i in range(self.number_of_generations):
            self.one_gen()
        return self.get_best_level()

    def get_best_level(self) -> Level:
        raise NotImplementedError("")
