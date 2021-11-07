import copy
from math import ceil
from typing import List
from matplotlib import pyplot as plt

import numpy as np
import scipy.stats
from baselines.ga.multi_pop_ga.multi_population_ga_pcg import MultiPopGAPCG, SingleElementFitnessFunction, SingleElementGAIndividual
from games.game import Game
from games.level import Level
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
from novelty_neat.novelty.distance_functions.distance import visual_diversity_normalised
from novelty_neat.novelty.novelty_metric import DistanceMetric, NoveltyArchive

class SparsenessFitnessFunction(SingleElementFitnessFunction):
    """
        This calculates the sparseness fitness function, i.e. the average distance between all pairs of items.
        The fitness is actually calculated as 1 / (desired_sparseness - sparse), normalised to between 0 and 1.


        Details in: M. Cook and S. Colton. Multi-faceted evolution of simple arcade games. In Computational Intelligence and Games (CIG), 2011 IEEE Conference on, pages 289–296, 2011
    """
    def __init__(self, desired_sparseness: float = 0, block_size: int = 10) -> None:
        super().__init__()
        self.desired_sparseness = desired_sparseness
        self.block_size = block_size
        
    def sparseness(self, one_indiv: SingleElementGAIndividual) -> float:
        """What this does is the following:
               We split the level up into chunks (Ai) of size `self.block_size`, and mean(compute sparse(Ai) for all i)
               sparse(Ai) is simply 2 * total / (n * n - 1), where n is the number of nonzero elements, and total is the 
               total pairwise distance (absolute difference between index) for each pair of non zero items

        Args:
            one_indiv (SingleElementGAIndividual): Single individual

        Returns:
            float: Sparseness
        """
        def sparse(array):
            pos = np.argwhere(array > 0)[:, 0]
            # with only one element, the sparseness is still 0.
            if len(pos) <= 1: return 0
            total = 0
            positions = pos
            n = len(positions)
            for i, p in enumerate(positions):
                for j in range(0, len(positions)):
                    # normalise distance to between 0 and 1.
                    dist = abs(p - positions[j]) / len(array)
                    total += dist
            return 2 * total / (n * n - 1)
        L = len(one_indiv.genome)
        nums = ceil(L / self.block_size)
        total_sparse = 0
        for i in range(nums):
            temp_arr = one_indiv.genome[i * self.block_size: (i+1)*self.block_size]
            total_sparse += sparse(temp_arr)
        return total_sparse / nums    

    def calc_fitness(self, individuals: List[SingleElementGAIndividual]) -> List[float]:
        X = [abs(self.sparseness(i) - self.desired_sparseness) for i in individuals]
        X = [1 / max(x, 0.1) / 10 for x in X]
        return X

class EntropyFitnessFunction(SingleElementFitnessFunction):
    """
        Calculates the Entropy fitness, similarly to the sparseness above
        We split the level up into chunks and calculate the average distance to desired entropy
        where entropy is the entropy of [x, y, z, ...] where x, y, z, etc are the proportion of that type of block.
    """
    def __init__(self, desired_entropy: float = 1, block_size: int = 114) -> None:
        super().__init__()
        self.desired_entropy = desired_entropy
        self.block_size = block_size
    
    def entropy(self, one_indiv: SingleElementGAIndividual) -> float:
        ans = np.array(one_indiv.genome)
        L = len(ans)
        nums = ceil(L / self.block_size)
        total = 0
        for i in range(nums):
            temp_arr = ans[i * self.block_size: (i+1)*self.block_size]
            counts = []
            for i in np.unique(temp_arr):
                counts.append((temp_arr == i).sum())
            ps = np.array(counts) / temp_arr.size
            e = scipy.stats.entropy(ps, base=2)
            if len(ps) >= 2:
                e /= abs(np.log2(len(ps)))
            assert -0.01 <= e <= 1.01, f"Entropy is invalid, {e}"
            total += e
        return e / nums


    def calc_fitness(self, individuals: List[SingleElementGAIndividual]) -> List[float]:
        X = [abs(self.entropy(i) - self.desired_entropy) for i in individuals]
        X = [1 / max(x, 0.1) / 10 for x in X]
        return X


class NoveltyFitnessFunctionSingleElement(SingleElementFitnessFunction):
    """ 
        Basically novelty metric for the single population fitness function.
    """
    def __init__(self, distance_function: DistanceMetric, max_dist: float, number_of_neighbours: int, lambd: int, archive_mode: NoveltyArchive):
        """See NoveltyMetric for more details

        Args:
            distance_function (DistanceMetric): This should give the distance between two arrays.
            max_dist (float): The maximum distance that can be achieved between two levels. This is used to normalise the distances between 0 and 1.
            number_of_neighbours (int, optional): The amount of closest neighbours to consider when calculating the novelty metric. Defaults to 10.
            lambd (int, optional): The number of individuals to add to the archive at each step. 

            archive_mode (NoveltyArchive, optional): How we choose which individuals need to get added. RANDOM chooses lambd random individuals, 
                and NOVEL chooses the lambd most novel individuals.
        """
        super().__init__()
        self.archive: List[SingleElementGAIndividual] = []
        self.previously_novel_individuals = None
        self.number_of_neighbours = number_of_neighbours
        self.lambd = lambd
        self.archive_mode = archive_mode
        self.distance_function = distance_function
        self.max_dist = max_dist

    def calc_fitness(self, individuals: List[SingleElementGAIndividual]) -> List[float]:
        assert self.number_of_neighbours < len(individuals), "Number of neighbours must be less than the number of levels"
        dist_matrix = np.zeros((len(individuals), len(individuals) + len(self.archive)))

        def dist(level1: SingleElementGAIndividual, level2: SingleElementGAIndividual) -> float:
            d = self.distance_function(level1.genome, level2.genome) / self.max_dist
            assert 0 <= d <= 1
            return d

        # Now calculate pairwise distance:
        for index1, level1 in enumerate(individuals):
            dist_matrix[index1, index1] = float('inf')
            for index2, level2 in list(enumerate(individuals))[index1+1:]:
                d = dist(level1, level2)
                dist_matrix[index1, index2] = d
                dist_matrix[index2, index2] = d
            
            # And from archive
            for index_archive, archived_level in enumerate(self.archive):
                d = dist(level1, archived_level)
                dist_matrix[index1, len(individuals) + index_archive] = d

        final_novelty_metrics = []
        # Now we need to calculate the closest K neighbours.
        for index, row in enumerate(dist_matrix):
            # Choose K closest neighbours
            row = sorted(row)[:self.number_of_neighbours]
            final_novelty_metrics.append(np.mean(row))  
        
        # Now add to archive if good enough, or randomly depending on the mode.

        indices = np.arange(len(individuals))
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
            copy.deepcopy(individuals[index]) for index in indices[:self.lambd]
        ])
        return final_novelty_metrics


    def __repr__(self) -> str:
        return f"NoveltyFitnessFunctionSingleElement(nneighbours={self.number_of_neighbours}, lambd={self.lambd}, mode={self.archive_mode})"

    def reset(self):
        self.archive = []
        return super().reset()


class CombinationFitnessFunctionSingleElement(SingleElementFitnessFunction):
    def __init__(self, fitnesses: List[SingleElementFitnessFunction], weights: List[int]) -> None:
        super().__init__()
        assert len(fitnesses) == len(weights)
        self.fitnesses = fitnesses
        self.weights = np.array(weights) / sum(weights)
    
    def calc_fitness(self, individuals: List[SingleElementGAIndividual]) -> List[float]:
        ans = 0
        for f, w in zip(self.fitnesses, self.weights):
            ans += np.array(f.calc_fitness(individuals)) * w
        return list(ans)
    
    def reset(self):
        for f in self.fitnesses:
            f.reset()

    
    

class MarioGAPCG(MultiPopGAPCG):
    """Mario GA PCG from:
        Ferreira, L., Pereira, L., & Toledo, C. (2014, July). A multi-population genetic algorithm for procedural generation of levels for platform games. In Proceedings of the Companion Publication of the 2014 Annual Conference on Genetic and Evolutionary Computation (pp. 45-46).
        We have separate populations to evolve the ground height, type of block, enemies and coin height.
        The first one is scored based on entropy and the others on sparseness.

    """
    def __init__(self, game: Game, init_level: Level, pop_size: int = 100, number_of_generations: int = 100,
                    desired_entropy: float = 0,
                    desired_sparseness_enemies: float = 0,
                    desired_sparseness_coins: float = 0.5,
                    desired_sparseness_blocks: float = 1,

                    entropy_block_size: int = 114,
                    enemies_block_size: int = 20,
                    coins_block_size: int = 10,
                    blocks_block_size: int = 10,

                    ground_maximum_height: int = 2,
                    coin_maximum_height: int = 2,

                    use_novelty: bool=False
                    ) -> None:
        """
        Args:
            game (Game): Game
            init_level (Level): The initial level
            pop_size (int, optional): Size of population. Defaults to 100.
            number_of_generations (int, optional): Number of gens to run. Defaults to 100.
            
            These desired values
                desired_entropy (for the ground) (float, optional): Defaults to 0.
                desired_sparseness_enemies (float, optional): Defaults to 0.
                desired_sparseness_coins (float, optional): Defaults to 0.5.
                desired_sparseness_blocks (float, optional): Defaults to 1.
            
            These block sizes control how large the blocks are for which entropy and sparseness is calculated.
                entropy_block_size (int, optional): Defaults to 114.
                enemies_block_size (int, optional): Defaults to 20.
                coins_block_size (int, optional): Defaults to 10.
                blocks_block_size (int, optional): Defaults to 10.
            
            The maximum values for the heights
                ground_maximum_height  (int, optional) . Defaults to 2
                coin_maximum_height    (int, optional) . Defaults to 2


            use_novelty (bool, optional). Uses novelty if this is true. Uses visual diversity. Defaults to False.
        """
        indiv_funcs = [
            # ground height, 0 means there is a gap
            lambda l: SingleElementGAIndividual(l, 0, ground_maximum_height, init=1),
            # enemies - types, either an enemy or not
            lambda l: SingleElementGAIndividual(l, 0, 1, init=0),
            # coins - heights: 0 means no coin there
            lambda l: SingleElementGAIndividual(l, 0, coin_maximum_height, init=0),
            # blocks - different types 0 is nothing, 1 is brick, 2 is question, 3 is tube
            lambda l: SingleElementGAIndividual(l, 0, 3, init=0),
        ]
        fitness_funcs = [
            EntropyFitnessFunction(desired_entropy, entropy_block_size), 
            SparsenessFitnessFunction(desired_sparseness_enemies, enemies_block_size), 
            SparsenessFitnessFunction(desired_sparseness_coins, coins_block_size), 
            SparsenessFitnessFunction(desired_sparseness_blocks, blocks_block_size)
        ]
        self.use_novelty = use_novelty
        if self.use_novelty:
            new_funcs = []
            for f in fitness_funcs:
                new_funcs.append(
                    CombinationFitnessFunctionSingleElement([
                        f, NoveltyFitnessFunctionSingleElement(visual_diversity_normalised, 1, 6, 1, NoveltyArchive.RANDOM)
                    ], [1, 1])
                )
            
            fitness_funcs = new_funcs

        self.level = init_level
        super().__init__(game, init_level, indiv_funcs, fitness_funcs, pop_size, number_of_generations)
    pass

    def get_best_level(self) -> Level:
        """Should only be called internally. Generates a level, after the evolution has taken place.
        """
        level = MarioLevel(self.level.width, self.level.height)
        # add in ground heights
        ground_height, has_enemy, coin_height, block_types = [self.populations[i][0].genome for i in range(len(self.populations))]
        H = level.map.shape[0]
        # Clear map
        level.map *= 0
        possible_block_types = ['brick', 'question', 'tube']
        for index, h in enumerate(ground_height):
            if h == 0: 
                continue
            level.map[H - h, index] = level.tile_types_reversed['solid']
            if has_enemy[index]:
                # Just above ground
                level.map[H - h - 1, index] = level.tile_types_reversed['enemy']
            if coin_height[index] != 0:
                ch = coin_height[index]
                level.map[H - h - ch, index] = level.tile_types_reversed['coin']
            block = block_types[index]
            if block != 0:
                which_block = possible_block_types[block - 1]
                block_height = np.random.randint(1, 5)
                level.map[H - h - block_height - 1, index] = level.tile_types_reversed[which_block]
        return level
    
if __name__ == '__main__':
    from timeit import default_timer as tmr
    args = dict(desired_entropy=0,
                        entropy_block_size=114,
                        ground_maximum_height=2,
                        
                        desired_sparseness_blocks=1,
                        blocks_block_size=10,


                        desired_sparseness_enemies=0.5,
                        enemies_block_size=20,

                        desired_sparseness_coins=0.5,
                        coin_maximum_height=2,
                        coins_block_size=10)
    temp = MarioGAPCG(MarioGame(MarioLevel()), MarioLevel(), 20, 10, **args)
    s = tmr()
    l = temp.generate_level()
    e = tmr()
    plt.imshow(l.show(False))
    plt.title(f"W = {l.map.shape}. Time = {e - s}")
    plt.show()

    temp = MarioGAPCG(MarioGame(MarioLevel()), MarioLevel(), 20, 10, use_novelty=True, **args)
    s = tmr()
    l = temp.generate_level()
    e = tmr()
    plt.imshow(l.show(False))
    plt.title(f"W = {l.map.shape}. Time = {e - s}. With novelty")
    plt.show()
