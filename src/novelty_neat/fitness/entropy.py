from math import ceil
from typing import List

import numpy as np
from games.level import Level
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from novelty_neat.fitness.fitness import NeatFitnessFunction
from novelty_neat.generation import NeatLevelGenerator
from novelty_neat.maze.neat_maze_level_generation import GenerateMazeLevelsUsingTiling
from novelty_neat.types import LevelNeuralNet
import scipy.stats as stats


class EntropyFitness(NeatFitnessFunction):
    def __init__(self, number_of_levels_to_generate: int, level_gen: NeatLevelGenerator, desired_entropy: int = None, subblock_size: int = 7, scale_to_one: bool = False):
        """This is the constructor for the entropy fitness function.

        Args:
            number_of_levels_to_generate (int): The number of levels to generate per network
            level_gen (NeatLevelGenerator): The method to generate levels with.
            desired_entropy (int, optional): The desired entropy level. Must be between 0 and 1, or None.
                If it is None, then the fitness will be the value of the entropy. Otherwise, it will be 1/abs(entropy - desired),
                capped at a maximum of 10. This is similar to the work of
                    Ferreira, L., Pereira, L., & Toledo, C. (2014, July). A multi-population genetic algorithm for procedural generation of levels for platform games. In Proceedings of the Companion Publication of the 2014 Annual Conference on Genetic and Evolutionary Computation (pp. 45-46).
                Defaults to None.

            subblock_size (int, optional): We split the level into non overlapping blocks of this size
                and assign the entropy to the average value of these blocks. For example, a level of size (14x14)
                and subblock size of 7 will result in 4 blocks.
                Defaults to 7.
            
            scale_to_one (bool, optional) If this is true, the entropy fitness ranges from 0 to 1 instead of from 1 to 10 as is the default
        """
        super().__init__(
            number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)
        self.desired_entropy = desired_entropy
        if desired_entropy is not None:
            assert 0 <= desired_entropy <= 1, f"Desired entropy {desired_entropy} is not in [0, 1]"
        self.subblock_size = subblock_size
        self.scale_to_one = scale_to_one

    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:
        def entropy_level(level: Level) -> float:
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
                e = stats.entropy(ps, base=2)
                return e

            map = level.map.astype(np.int32)
            size = self.subblock_size
            numx = ceil(map.shape[1] / size)
            numy = ceil(map.shape[0] / size)
            total_entropies = []
            for xi in range(numx-1):
                for yi in range(numy-1):
                    array = map[yi * size:(yi+1)*size, xi * size:(xi+1)*size]
                    total_entropies.append(entropy(array))
            return np.mean(total_entropies)

        final_answer = []
        for level_group in levels:
            total = 0
            for level in level_group:
                entropy_for_this_level = entropy_level(level)
                # Now check if our desired entropy is None, in which case we simply return the normal value of entropy.
                if self.desired_entropy is None:
                    total += entropy_for_this_level
                else:
                    # Otherwise it's just distance.
                    # Distance is the same as absolute value in one dimension
                    to_add = 1 / max(0.1, abs(entropy_for_this_level - self.desired_entropy))
                    # If we should scale the results to between 0 and 1.
                    if self.scale_to_one:
                        to_add /= 10 

                    total += to_add
                    
            final_answer.append(total / len(level_group))
        return final_answer

    def __repr__(self) -> str:
        return f"EntropyFitness(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen}, desired_entropy={self.desired_entropy}, subblock_size={self.subblock_size})"


if __name__ == '__main__':
    func = EntropyFitness(
        5, GenerateMazeLevelsUsingTiling(game=MazeGame(MazeLevel())))
    print(func.params())
