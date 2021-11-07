import copy
from math import ceil
from typing import AnyStr, Callable, Dict, List, Tuple
from matplotlib import pyplot as plt

import numpy as np
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

class GeneticAlgorithmIndividual:
    """A single individual
    """
    def __init__(self, level: Level):
        self.level = level
        self.fitness = 0

    def mutate(self, p=0.05):
        """Mutates this individual randomly, toggling a tile

        Args:
            p (float, optional): The probability of mutation, i.e. p * len(map) gives amount of tiles that will change (roughly). 
            Defaults to 0.05.
        """
        raise NotImplementedError()

    def crossover(self, other: "GeneticAlgorithmIndividual") -> Tuple["GeneticAlgorithmIndividual", "GeneticAlgorithmIndividual"]:
        """Performs crossover between these two parents.
            Tuple["GeneticAlgorithmIndividual", "GeneticAlgorithmIndividual"]: The two children
        """
        raise NotImplementedError()

    def calc_fitness(self) -> float:
        """Calculates the fitness.
        Returns:
            float: The fitness
        """
        return 1

class GeneticAlgorithmIndividualMaze(GeneticAlgorithmIndividual):
    """
        A single individual for the Maze Game.
    """
    def __init__(self, level: Level, desired_entropy: float, entropy_weight=15):
        self.level = level
        self.fitness = 0
        self.desired_entropy = desired_entropy
        self.entropy_weight = entropy_weight

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

    def crossover(self, other: "GeneticAlgorithmIndividual") -> Tuple["GeneticAlgorithmIndividual", "GeneticAlgorithmIndividual"]:
        """Performs simple 2 point crossover between these two individuals to generate another one.
        If p1 = [a, b, c, d, e, f] and
           p2 = [a1, b1, c1, d1, e1, f1]
           the children will be (for example)
           c1 = [a, b, c1, d1, e, f]
           c2 = [a1, b1, c, d, e1, f1]
        
        The 2d map is flattened into a 1-d array that is then used to perform the crossover
        Returns:
            Tuple["GeneticAlgorithmIndividual", "GeneticAlgorithmIndividual"]: The two children
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
            GeneticAlgorithmIndividualMaze(Level(self.level.width, self.level.height, self.level.tile_types,
                                             map=vec.reshape(self.level.height, self.level.width)), desired_entropy=self.desired_entropy)
            for vec in children
        ]
        return children

    def entropy(self) -> float:
        """Calculates the entropy of the level. Using
            - (p(0) * log(p(0)) + p(1) * log(p(1)))
        where p(0) and p(1) is the probability of the map having a 0 or 1 respectively (i.e. (map ==0).sum() / map.size)

        We actually split the map up into square grids (4x4) and calculate the entropy in each of those.
        Returns:
            float: The entropy.
        """
        def entropy(array):
            counts = []
            for i in [0, 1]:
                counts.append((array == i).sum())
            ps = np.array(counts) / array.size
            e = -(ps * np.log2(ps)).sum()
            return e

        map = self.level.map.astype(np.int32)
        size = 7
        numx = ceil(map.shape[1] / size)
        numy = ceil(map.shape[0] / size)
        total_entropies = []
        for xi in range(numx-1):
            for yi in range(numy-1):
                array = map[yi * size:(yi+1)*size, xi * size:(xi+1)*size]
                total_entropies.append(entropy(array))
        return np.mean(total_entropies)
        # Global entropy
        return entropy(map)

    def calc_fitness(self) -> float:
        """Calculates the fitness. At the moment it is a bit hardcoded to check if the maze has connected beginning and end, 
           as well empty initial and ending states.

        Returns:
            float: The fitness
        """
        # return self.entropy()
        map = self.level.map.astype(np.int32)
        score = 100
        # Penalise if start or end is not empty
        if map[0, 0] == 1:
            score -= 49
        if map[-1, -1] == 1:
            score -= 49
        connected = morph.label(map+1, connectivity=1)
        # If path between start and end, then +
        if connected[0, 0] == connected[-1, -1]:
            score += 50

        entropy = self.entropy()
        dist = max(0.1, abs(entropy - self.desired_entropy))
        return score + self.entropy_weight * 1/dist


class GeneticAlgorithmPCG(PCGMethod):
    def __init__(self, game: Game, init_level: Level, individual_func: Callable[[Level], GeneticAlgorithmIndividual], population_size: int = 50, number_of_generations: int = 100) -> None:
        super().__init__(game, init_level)
        self.population_size = population_size
        self.individual_func = individual_func
        self.level = init_level
        self.number_of_generations = number_of_generations
        self.reset()
        self.logger: Logger = None

    def reset(self):
        self.population: List[GeneticAlgorithmIndividual] = [self.individual_func(copy.deepcopy(self.level)) for i in range(self.population_size)]
        self.gen_count = 0
        self.best_agent = None

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
            a1 = np.random.choice(self.population, p=probs)
            a2 = np.random.choice(self.population, p=probs)
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
        for a in self.population:
            a.fitness = a.calc_fitness()
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
    def train(self, logger: Logger) -> List[Dict[str, AnyStr]]:
        self.logger = logger
        return super().train(logger)

if __name__ == "__main__":
    level = MazeLevel(40, 40)
    level.map = (np.random.rand(*level.map.shape) > 0.5)
    alg = GeneticAlgorithmPCG(Game(level), level, lambda l: GeneticAlgorithmIndividualMaze(l, 1), 50)
    for i in range(400):
        alg.one_gen()
    best_level = alg.best_agent.level
    plt.imshow(1-best_level.map, cmap='gray'); plt.show()