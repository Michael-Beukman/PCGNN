from functools import partial

from matplotlib import pyplot as plt
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from novelty_neat.fitness.fitness import NeatFitnessFunction
from novelty_neat.generation import NeatLevelGenerator
from novelty_neat.maze.utils import path_length
from novelty_neat.novelty_neat import LevelNeuralNet, NoveltyNeatPCG
import os
import pickle
from common.methods.pcg_method import PCGMethod
from typing import Any, Dict, List, Callable, Tuple, Union
from common.utils import get_date
from games.game import Game
from games.level import Level
from experiments.logger import Logger
import neat
import numpy as np
from novelty_neat.maze.neat_maze_level_generation import GenerateMazeLevelUsingOnePass, GenerateMazeLevelsUsingTiling
from baselines.ga.genetic_algorithm_pcg import GeneticAlgorithmIndividualMaze
import cProfile
import skimage.morphology as morph
class SolvabilityFitness(NeatFitnessFunction):
    """Returns a solvability score based on if the maze level is solvable or not. 1 for solvable and 0 for unsolvable.
    """
    def __init__(self, number_of_levels_to_generate: int, level_gen: NeatLevelGenerator):
        super().__init__(number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)
    
    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:
        answer = []
        for level_group in levels:
            ans = 0
            for level in level_group:
                ans += (1 if path_length(level, first_tile=False) > 0 else 0)
            answer.append(ans / len(level_group))
        return answer

class PartialSolvabilityFitness(NeatFitnessFunction):
    """Returns a partial solvability score, namely, we take into account:
        - if the initial cell is empty or not
        - if the goal cell is empty or not
        - if the initial cell and goal cell is connected.

        the score is 1 if all three conditions are met, and 0 if none are met.
    """
    def __init__(self, number_of_levels_to_generate: int, level_gen: NeatLevelGenerator):
        super().__init__(number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)
    
    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:
        def get_value_for_level(level: Level):
            map = level.map
            score = 100
            # Penalise if start or end is not empty
            if map[0, 0] == 1:
                score -= 50
            if map[-1, -1] == 1:
                score -= 50
            connected = morph.label(map+1, connectivity=1)
            
            # If path between start and end, then +50
            if connected[0, 0] == connected[-1, -1] and map[0, 0] == 0:
                score += 50
            
            return score / 150

        answer = []
        for level_group in levels:
            ans = 0
            for level in level_group:
                ans += get_value_for_level(level)
            answer.append(ans / len(level_group))
        return answer

def _get_path_lengths_maze(levels: List[MazeLevel]) -> List[float]:
    return [path_length(l) for l in levels]

class PathLengthFitness(NeatFitnessFunction):
    """This simply returns the path length / num_tiles as a fitness function
    """
    def __init__(self, number_of_levels_to_generate: int, level_gen: NeatLevelGenerator, should_reward_larger_levels_more=False):
        super().__init__(number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)
        self.should_reward_larger_levels_more = should_reward_larger_levels_more
    
    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:
        ans = []
        for group in levels:
            w, h = group[0].width, group[0].height
            lengths = np.array(_get_path_lengths_maze(group))
            if self.should_reward_larger_levels_more:
                # Reward levels that don't just have a simple path more.
                lengths[lengths > w + h] = lengths[lengths > w + h] * 4
            lengths = np.clip(lengths, 0, group[0].map.size)
            ans.append(np.mean(lengths) / group[0].map.size)
        return ans
    def __repr__(self) -> str:
        return f"PathLengthFitness(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen}, should_reward_larger_levels_more={self.should_reward_larger_levels_more})"