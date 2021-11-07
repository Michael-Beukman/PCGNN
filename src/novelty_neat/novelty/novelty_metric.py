# from typing import Any, Dict, List, Callable, Tuple, Union

import copy
from enum import Enum
from turtle import distance
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from games.level import Level
from games.mario.mario_level import MarioLevel
from games.maze.maze_level import MazeLevel
from novelty_neat.fitness.fitness import NeatFitnessFunction
from novelty_neat.generation import NeatLevelGenerator
from novelty_neat.mario.utils import get_path_trajectory_mario
from novelty_neat.maze.neat_maze_fitness import _get_path_lengths_maze
from novelty_neat.maze.utils import get_path_trajectory, shortest_path
from novelty_neat.novelty_neat import LevelNeuralNet

DistanceMetric = Callable[[np.ndarray, np.ndarray], float]

# it takes (positions, actions, map) from each level.
TrajectoryDistanceMetric = Callable[[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]], float]


class NoveltyArchive(Enum):
    NOVEL = 1
    RANDOM = 2


class NoveltyMetric(NeatFitnessFunction):
    """
        The general novelty metric as described in
            Joel Lehman, Kenneth O. Stanley; Abandoning Objectives: Evolution Through the Search for Novelty Alone. Evol Comput 2011; 19 (2): 189–223. doi: https://doi.org/10.1162/EVCO_a_00025

        This class needs to be provided by multiple different problem specific functions, like generating a level from a network,
        as well as a game-specific distance metric.

        There are also parameters, such as how many of the closest neighbours to consider when calculating the novelty metric itself.


        We changed the implementation to more closely follow 
            Gomes, J., Mariano, P., & Christensen, A. L. (2015, July). Devising effective novelty search algorithms: A comprehensive empirical study. In Proceedings of the 2015 Annual Conference on Genetic and Evolutionary Computation (pp. 943-950).
        because in the fact that we use lambda to control how many new genomes get added to the archive of previous individuals.
        We will have two modes, 'random' in which lambda random genomes get added each generation or 'novel', where the lambda indivs with the highest 
        novelty will get added
    """

    def __init__(self, level_generator: NeatLevelGenerator, 
                 distance_function: Union[DistanceMetric, TrajectoryDistanceMetric],
                 max_dist: float,
                 number_of_levels: int = 10,
                 number_of_neighbours: int = 10,
                 lambd: int = 2,
                 archive_mode: NoveltyArchive = NoveltyArchive.RANDOM,
                 should_use_all_pairs: bool = False,
                 distance_mode: str = "distance"
                 ):
        """Initialises this novelty calculator.

        Args:
            level_generator (NeatLevelGenerator): This should take in a network and some random numbers and generate a level
            distance_function (Union[DistanceMetric, TrajectoryDistanceMetric]): This should give the distance between two levels, 
                    either between their maps, or between their trajectories.
            max_dist (float): The maximum distance that can be achieved between two levels. This is used to normalise the distances between 0 and 1.
            number_of_levels (int, optional): How many levels count as a representative sample of the network. Defaults to 10.
            number_of_neighbours (int, optional): The amount of closest neighbours to consider when calculating the novelty metric. Defaults to 10.
            lambd (int, optional): The number of individuals to add to the archive at each step. Defaults to 2

            archive_mode (NoveltyArchive, optional): How we choose which individuals need to get added. RANDOM chooses lambd random individuals, 
                and NOVEL chooses the lambd most novel individuals.

            should_use_all_pairs (bool, optional): If this is True, then in the calculation of novelty of generators A B, we don't simply just compare
                A[i] and B[i], but we rather compare A[i] with B[j] for all pairs i, j. This will be slower in general, but might 
                avoid some issues due to the ordering. Defaults to False
            
            distance_mode (str, optional). Either 'distance' or 'trajectory' depending on which function to use. Defaults to "distance"

        """
        super().__init__(number_of_levels, level_generator)
        self.distance_function = distance_function
        self.number_of_neighbours = number_of_neighbours
        self.lambd = lambd
        self.archive_mode = archive_mode
        self.previously_novel_individuals: List[LevelNeuralNet] = []
        self.max_dist = max_dist
        self.should_use_all_pairs = should_use_all_pairs
        self.distance_mode = distance_mode
        assert self.distance_mode in ['distance', 'trajectory'], "Distance mode must be in ['distance', 'trajectory']"

    def _get_path_trajectory(self, levels: List[Level]) -> List[List[Tuple[int, int]]]:
        if isinstance(levels[0], MazeLevel):
            return [get_path_trajectory(l) for l in levels]
        elif isinstance(levels[0], MarioLevel):
            return [get_path_trajectory_mario(l) for l in levels]
        else:
            assert 1 == 0, "Bad Level Type"

    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:
        """This actually calculates the novelty metric for each of these given networks.
            It first of all generates self.number_of_levels levels from each network. 
            It uses these levels to compute the pairwise distance between all the networks, as well as the 
            archive of previously novel individuals. The novelty metric for a single network is 
            mean( distance of closes `self.number_of_neighbours` other networks).
            It also adds genomes to the archive of individuals according to the self.lamdb and self.archive_mode as described above.


            Each value in the returned list will be between 0 and 1, with 0 indicating a low level of novelty and 1 indicating maximum novelty.
        Args:
            nets (List[LevelNeuralNet]): 

        Returns:
            List[float]:
        """
        def my_dist_all_pairs(levels1: List[Level], levels2: List[Level]):
            if self.distance_mode == 'trajectory':
                paths1 = self._get_path_trajectory(levels1)
                paths2 = self._get_path_trajectory(levels2)
            # Computes the distance for all pairs.
            ans = 0
            N = 0
            for i in range(len(levels1)):
                for j in range(i+1, len(levels1)):
                    if self.distance_mode == 'trajectory':
                        ans += self.distance_function((np.array(paths1[i] or []), np.array([]), levels1[i].map), 
                                                     (np.array(paths2[j] or []), np.array([]), levels2[j].map))
                    else:
                        ans += self.distance_function(levels1[i].map, levels2[j].map)
                    N += 1

            # Normalise it to be between 0 and 1.
            ans = ans / self.max_dist / N
            assert 0 <= ans <= 1
            return ans

        def my_dist(levels1: List[Level], levels2: List[Level]):
            if self.should_use_all_pairs:
                return my_dist_all_pairs(levels1, levels2)
            if self.distance_mode == 'trajectory':
                paths1 = self._get_path_trajectory(levels1)
                paths2 = self._get_path_trajectory(levels2)
            ans = 0
            N = 0
            for i, (l1, l2) in enumerate(zip(levels1, levels2)):
                if self.distance_mode == 'trajectory':
                    ans += self.distance_function((np.array(paths1[i] or []), np.array([]), levels1[i].map), 
                                                  (np.array(paths2[i] or []), np.array([]), levels2[i].map))
                else:
                    ans += self.distance_function(l1.map, l2.map)
                N += 1
            # Normalise it to be between 0 and 1.
            ans = ans / self.max_dist / N
            assert 0 <= ans <= 1
            return ans
        # If we already generated enough levels, then we can just use those.
        # Otherwise we should use others.
        if len(levels) and len(levels[0]) == self.number_of_levels:
            mylevels = levels
        else:
            mylevels = []
            for net in nets:
                temp = []
                for i in range(self.number_of_levels):
                    temp.append(self.level_gen(net))
                mylevels.append(temp)

        archive_levels = []
        for net in self.previously_novel_individuals:
            temp = []
            for i in range(self.number_of_levels):
                temp.append(self.level_gen(net))
            archive_levels.append(temp)

        # Generate a pairwise distance for each net.
        dist_matrix = np.zeros(
            (len(mylevels), len(mylevels) + len(self.previously_novel_individuals)))
        for index1, levels1 in enumerate(mylevels):
            dist_matrix[index1, index1] = float('inf')
            for index2, levels2 in list(enumerate(mylevels))[index1+1:]:
                d = my_dist(levels1, levels2)
                dist_matrix[index1, index2] = d
                dist_matrix[index2, index2] = d

            for index_archive, archive_ls in enumerate(archive_levels):
                d = my_dist(levels1, archive_ls)
                dist_matrix[index1, len(mylevels) + index_archive] = d

        final_novelty_metrics = []
        # Now we need to calculate the closest K neighbours.
        for index, row in enumerate(dist_matrix):
            # Choose K closest neighbours
            row = sorted(row)[:self.number_of_neighbours]
            final_novelty_metrics.append(np.mean(row))

        # Now add to archive if good enough
        indices = np.arange(len(nets))
        if self.archive_mode == NoveltyArchive.RANDOM:
            # Add in self.lambd random nets.
            # Shuffle
            np.random.shuffle(indices)
            # Add in random 2
            self.previously_novel_individuals.extend([
                copy.deepcopy(nets[index]) for index in indices[:self.lambd]
            ])
        elif self.archive_mode == NoveltyArchive.NOVEL:
            # Most novel individuals
            sorted_list = sorted(
                zip(final_novelty_metrics, indices), reverse=True)
            self.previously_novel_individuals.extend([
                copy.deepcopy(nets[index]) for score, index in sorted_list[:self.lambd]
            ])
        else:
            raise Exception(
                f"{self.archive_mode} is not a valid NovelArchive mode")
        return final_novelty_metrics

    def params(self) -> Dict[str, Any]:
        dic = super().params()
        dic['distance'] = self.distance_function.__name__
        dic['number_of_neighbours'] = self.number_of_neighbours
        dic['lambd'] = self.lambd
        dic['archive_mode'] = str(self.archive_mode)
        dic['max_dist'] = self.max_dist
        dic['should_use_all_pairs'] = self.should_use_all_pairs
        return dic

    def __repr__(self) -> str:
        return f"NoveltyMetric({self.level_gen}, {self.distance_function.__name__}, {self.max_dist}, {self.number_of_levels}, {self.number_of_neighbours}, {self.lambd}, {self.archive_mode}, should_use_all_pairs={self.should_use_all_pairs}, distance_mode={self.distance_mode})"


class NoveltyIntraGenerator(NeatFitnessFunction):
    """This is a novelty metric that measures the novelty between levels in the same generator.
    Thus, if we have levels:
       [[l11, l12, l13, ..., l1n], [l21, l22, l23, ..., l2n], ...]
       Then this only measures novelty between l1i, and l1j -> i.e. levels from the same generator.


       We measure the novelty for each level, without any previous archive, and then assign the average to the network.
    """

    def __init__(self, number_of_levels_to_generate: int, level_gen: NeatLevelGenerator,
                 distance_function: DistanceMetric,
                 max_dist: float,
                 number_of_neighbours: int = 10,
                 ):
        """
        Args:
            number_of_levels_to_generate (int): The number of levels to generate for each network.
            level_gen (NeatLevelGenerator): How to generate levels.
            distance_function (DistanceMetric): How to calculate distance.
            max_dist (float): The maximum possible distance. Used to normalise the output of distance_function
            number_of_neighbours (int, optional): How many neighbours to use in the novelty function. Defaults to 10.
        """
        super().__init__(
            number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)
        self.distance_function = distance_function
        self.max_dist = max_dist
        self.number_of_neighbours = number_of_neighbours

        assert self.number_of_neighbours < self.number_of_levels, "Number of neighbours to choose has to be strictly less than number of levels to generate"

    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:
        if not (len(levels) and len(levels[0]) == self.number_of_levels):
            levels = self.get_levels(nets)

        final_answer = []
        dist_matrix = np.zeros(
            (self.number_of_levels, self.number_of_levels), dtype=np.float32)
        for level_group in levels:
            # Clear matrix
            dist_matrix.fill(0.0)
            for index1, level1 in enumerate(level_group):
                dist_matrix[index1, index1] = float('inf')
                for index2, level2 in list(enumerate(level_group))[index1 + 1:]:
                    d = self.distance_function(
                        level1.map, level2.map) / self.max_dist
                    dist_matrix[index1, index2] = d
            total = 0

            for index, row in enumerate(dist_matrix):
                # Choose K closest neighbours
                row = sorted(row)[:self.number_of_neighbours]
                total += np.mean(row)
            final_answer.append(total / self.number_of_levels)

        return final_answer

    def params(self) -> Dict[str, Any]:
        dic = super().params()
        dic['distance'] = self.distance_function.__name__
        dic['number_of_neighbours'] = self.number_of_neighbours
        dic['max_dist'] = self.max_dist
        return dic

    def __repr__(self) -> str:
        return f"NoveltyIntraGenerator({self.number_of_levels}, {self.level_gen}, {self.distance_function.__name__}, {self.max_dist}, {self.number_of_neighbours})"


class NoveltyMetricDirectGA(NoveltyMetric):
    """This is a modification of the above novelty metric, for direct genetic algorithms.
            It does not use the networks, and performs some assertions to ensure that it is actually used for a GA.
            In its archive, it stores the novel levels now instead of the networks. 
    """
    def __init__(self, distance_function: DistanceMetric, max_dist: float, number_of_levels: int, number_of_neighbours: int, lambd: int, archive_mode: NoveltyArchive):
        """See NoveltyMetric for more details

        Args:
            distance_function (DistanceMetric): This should give the distance between two levels.
            max_dist (float): The maximum distance that can be achieved between two levels. This is used to normalise the distances between 0 and 1.
            number_of_levels (int, optional): How many levels count as a representative sample of the network. Defaults to 10.
            number_of_neighbours (int, optional): The amount of closest neighbours to consider when calculating the novelty metric. Defaults to 10.
            lambd (int, optional): The number of individuals to add to the archive at each step. Defaults to 2

            archive_mode (NoveltyArchive, optional): How we choose which individuals need to get added. RANDOM chooses lambd random individuals, 
                and NOVEL chooses the lambd most novel individuals.
        """
        super().__init__(None, distance_function, max_dist, number_of_levels=number_of_levels,
                         number_of_neighbours=number_of_neighbours, lambd=lambd, archive_mode=archive_mode, should_use_all_pairs=False)
        self.archive: List[Level] = []
        assert number_of_levels == 1, "Number of levels for Direct GA Novelty must be 1"
        self.previously_novel_individuals = None

    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:
        assert nets is None, "nets must be None for the Direct GA fitness"
        levels_flat = []
        for l in levels:
            assert len(l) == 1
            levels_flat += l
        assert self.number_of_neighbours < len(levels_flat), "Number of neighbours must be less than the number of levels"
        levels = levels_flat
        dist_matrix = np.zeros((len(levels), len(levels) + len(self.archive)))

        def dist(level1: Level, level2: Level) -> float:
            d = self.distance_function(level1.map, level2.map) / self.max_dist
            assert 0 <= d <= 1
            return d

        # Now calculate pairwise distance:
        for index1, level1 in enumerate(levels):
            dist_matrix[index1, index1] = float('inf')
            for index2, level2 in list(enumerate(levels))[index1+1:]:
                d = dist(level1, level2)
                dist_matrix[index1, index2] = d
                dist_matrix[index2, index2] = d
            
            # And from archive
            for index_archive, archived_level in enumerate(self.archive):
                d = dist(level1, archived_level)
                dist_matrix[index1, len(levels) + index_archive] = d

        final_novelty_metrics = []
        # Now we need to calculate the closest K neighbours.
        for index, row in enumerate(dist_matrix):
            # Choose K closest neighbours
            row = sorted(row)[:self.number_of_neighbours]
            final_novelty_metrics.append(np.mean(row))  
        
        # Now add to archive if good enough, or randomly depending on the mode.

        indices = np.arange(len(levels))
        if self.archive_mode == NoveltyArchive.RANDOM:
            # Shuffle
            np.random.shuffle(indices)

        elif self.archive_mode == NoveltyArchive.NOVEL:
            # Most novel individuals
            sorted_list = sorted(zip(final_novelty_metrics, indices), reverse=True)
            indices = [index for score, index in sorted_list]
        else:
            raise Exception(
                f"{self.archive_mode} is not a valid NovelArchive mode")
        
        self.archive.extend([
            copy.deepcopy(levels[index]) for index in indices[:self.lambd]
        ])
        return final_novelty_metrics


    def __repr__(self) -> str:
        return f"NoveltyMetricDirectGA({self.level_gen}, {self.distance_function.__name__}, {self.max_dist}, {self.number_of_levels}, {self.number_of_neighbours}, {self.lambd}, {self.archive_mode}, should_use_all_pairs={self.should_use_all_pairs})"

    def params(self) -> Dict[str, Any]:
        ps = super().params()
        ps['method'] = 'DirectGA'
        return ps
    
    def reset(self) -> None:
        self.archive = []


if __name__ == '__main__':
    def euclidean_distance(a, b):
        return np.linalg.norm(a - b)

    f = euclidean_distance
    print(f.__name__)
    a = np.zeros((14, 14))
    b = np.ones((14, 14))
    a = (np.random.rand(14, 14) > 0.5) * 1.0
    b = (np.random.rand(14, 14) > 0.5) * 1.0
    print(euclidean_distance(a, b))
