import copy
from math import ceil
from typing import Any, Callable, Dict, List, Tuple
from matplotlib import pyplot as plt

import numpy as np
from baselines.ga.direct_ga_fitness import DirectFitness
from common.methods.pcg_method import PCGMethod
from experiments.logger import Logger
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

class _GeneralGAIndividual:
    """A single individual
    """
    def __init__(self, level: Level):
        self.level = level
        self.fitness = 0

    def mutate(self, p=0.05) -> None:
        """Mutates this individual randomly, toggling a tile

        Args:
            p (float, optional): The probability of mutation, i.e. p * len(map) gives amount of tiles that will change (roughly). 
            Defaults to 0.05.
        """
        raise NotImplementedError()

    def crossover(self, other: "_GeneralGAIndividual") -> Tuple["_GeneralGAIndividual", "_GeneralGAIndividual"]:
        """Performs crossover between these two parents.
            Tuple["_GeneralGAIndividual", "_GeneralGAIndividual"]: The two children
        """
        raise NotImplementedError()

class _GeneralGAIndividualMaze(_GeneralGAIndividual):
    """
        A single individual for the Maze Game.
    """
    def __init__(self, level: Level):
        super().__init__(level)

    def mutate(self, p=0.05):
        """Mutates this individual randomly, toggling a tile

        Args:
            p (float, optional): The probability of mutation, i.e. p * len(map) gives amount of tiles that will change (roughly). 
            Defaults to 0.05.
        """
        should_change = np.random.rand(*self.level.map.shape) < p
        coords = np.argwhere(should_change == 1)
        for row, col in coords:
            currently_located = self.level.map[row, col]
            others = [k for k in self.level.tile_types if k != currently_located]
            self.level.map[row, col] = np.random.choice(others)

    def crossover(self, other: "_GeneralGAIndividual") -> Tuple["_GeneralGAIndividual", "_GeneralGAIndividual"]:
        """Performs simple 2 point crossover between these two individuals to generate another one.
        If p1 = [a, b, c, d, e, f] and
           p2 = [a1, b1, c1, d1, e1, f1]
           the children will be (for example)
           c1 = [a, b, c1, d1, e, f]
           c2 = [a1, b1, c, d, e1, f1]
        
        The 2d map is flattened into a 1-d array that is then used to perform the crossover
        Returns:
            Tuple["_GeneralGAIndividual", "_GeneralGAIndividual"]: The two children
        """
        v1 = self.level.map.reshape(-1, 1)
        v2 = other.level.map.reshape(-1, 1)
        i1 = np.random.randint(0, len(v1))
        i2 = np.random.randint(0, len(v1))
        if i2 < i1:
            i1, i2 = i2, i1

        children = [np.zeros_like(v1) for i in range(2)]
        parents = [v1, v2]
        for i, c in enumerate(children):
            c[:i1] = parents[i][:i1]
            c[i1:i2] = parents[1-i][i1:i2]
            c[i2:] = parents[i][i2:]
        children = [
            _GeneralGAIndividualMaze(Level(self.level.width, self.level.height, self.level.tile_types,
                                             map=vec.reshape(self.level.height, self.level.width)))
            for vec in children
        ]
        return children


    
class GeneralGAPCG(PCGMethod):
    """This attempts to be a general genetic algorithm that is somewhat independent of whatever fitness metrics to use.

    Args:
        PCGMethod ([type]): [description]
    """
    def __init__(self, game: Game, init_level: Level, individual_func: Callable[[Level], _GeneralGAIndividual], fitness_function: DirectFitness, population_size: int = 50, number_of_generations: int = 100) -> None:
        super().__init__(game, init_level)
        self.population_size = population_size
        self.individual_func = individual_func
        self.level = init_level
        self.number_of_generations = number_of_generations
        self.fitness_function = fitness_function
        self.reset()
        self.logger: Logger = None

    def reset(self):
        self.population: List[_GeneralGAIndividual] = [self.individual_func(copy.deepcopy(self.level)) for i in range(self.population_size)]
        self.gen_count = 0
        self.best_agent = None
        self.fitness_function.reset()

    def one_gen(self):
        """Performs one generation, evaluates and breeds.
        """
        probs = self.evaluate()
        self.breed(probs)
        self.gen_count += 1

    def breed(self, probs: List[float]):
        """Breeds the current population to form the next one. 
           The individual with the best score is kept unchanged, and the others are from crossover with parents.

        Args:
            probs (List[float]): With what probability should each parent be chosen.
        """
        best_agent = np.argmax(probs)
        self.best_agent = self.population[best_agent]
        new_pop = [self.best_agent]

        while len(new_pop) < self.population_size:
            a1: _GeneralGAIndividual = np.random.choice(self.population, p=probs)
            a2: _GeneralGAIndividual = np.random.choice(self.population, p=probs)
            c = 0
            while a2 == a1 and c < 5:
                c += 1
                a2 = np.random.choice(self.population, p=probs)

            if (a1 == a2):
                print(" # ", end='')
            child1, child2 = a1.crossover(a2)
            child1.mutate(0.2)
            child2.mutate(0.2)
            new_pop.append(child1)
            if len(new_pop) < self.population_size:
                new_pop.append(child2)
        self.population = new_pop

    def evaluate(self) -> List[float]:
        """This simply goes through the population and calculates their fitnesses.
           It normalises these fitnesses, so that sum(evaluate()) == 1.

        Returns:
            List[float]: Probability of each parent to be chosen for breeding.
        """
        total_fit = 0
        max_fit = -1
        fitnesses = self.fitness_function.calc_fitness(
            [i.level for i in self.population], self.logger
        )
        for a, f in zip(self.population, fitnesses):
            a.fitness = f
            max_fit = max(max_fit, a.fitness)
            total_fit += a.fitness

        self.best_fit = max_fit
        probs = [
            a.fitness / max(1, total_fit) for a in self.population
        ]
        return probs

    def generate_level(self) -> Level:
        self.reset()
        for i in range(self.number_of_generations):
            self.one_gen()
        return self.best_agent.level
    
    def train(self, logger: Logger) -> List[Dict[str, Any]]:
        self.logger = logger
        return super().train(logger)
    

if __name__ == "__main__":
    level = MazeLevel()
    level.map = (np.random.rand(*level.map.shape) > 0.5)
    level_generator = None
    num_levels = 0

    fit = DirectFitness(CombinedFitness([
                                    NoveltyMetric(level_generator, visual_diversity, level.width * level.height, 2, number_of_levels=1, number_of_neighbours=25, pmin=10), 
                                    EntropyFitness(1, level_generator), 
                                    SolvabilityFitness(num_levels, level_generator),
                                    ], [1, 1, 1],
                                  number_of_levels_to_generate=num_levels, level_gen=level_generator))

    alg = GeneralGAPCG(Game(level), level, lambda l: _GeneralGAIndividualMaze(l), fitness_function=fit, population_size=50)
    for i in range(20):
        alg.one_gen()
    best_level = alg.best_agent.level
    plt.imshow(1-best_level.map, cmap='gray')
    # plt.imshow(morph.label(alg.best_agent.level.map + 1))
    metric = LeniencyMetric(MazeGame(level))
    print("Leniency = ", metric.evaluate([best_level]))
    level2 = np.zeros_like(best_level.map) + 1
    level2[0, :] = 0
    level2[:, -1] = 0
    l = MazeLevel(); l.map = level2
    print("Leniency 2 = ", metric.evaluate([l]))
    plt.show()
    print(alg.best_agent.level.map)