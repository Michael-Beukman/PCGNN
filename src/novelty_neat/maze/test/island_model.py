import os
import pickle
from math import ceil
from typing import List, Tuple

import neat
import numpy as np
import ray
import wandb
from common.types import Verbosity
from common.utils import get_date
from experiments.config import Config
from experiments.experiment import Experiment
from experiments.logger import NoneLogger, PrintLogger
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from matplotlib import pyplot as plt
from metrics.combination_metrics import RLAndSolvabilityMetric
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.horn.linearity import LinearityMetric
from metrics.rl.tabular.rl_agent_metric import RLAgentMetric
from metrics.rl.tabular.rl_difficulty_metric import RLDifficultyMetric
from metrics.solvability import SolvabilityMetric
from novelty_neat.fitness.fitness import CombinedFitness, NeatFitnessFunction
from novelty_neat.maze.neat_maze_fitness import (PathLengthFitness,
                                                 SolvabilityFitness)
from novelty_neat.maze.neat_maze_level_generation import \
    GenerateMazeLevelsUsingTiling
from novelty_neat.maze.utils import path_length
from novelty_neat.novelty.distance_functions.distance import \
    visual_diversity_only_reachable
from novelty_neat.novelty.novelty_metric import (NoveltyArchive,
                                                 NoveltyIntraGenerator,
                                                 NoveltyMetric)
from novelty_neat.novelty_neat import NoveltyNeatPCG


def run_experiments_test():
    ray.init()
    name = 'v11_test_neat_maze_island'
    game = 'Maze'
    method = 'NeatNovelty'
    date = get_date()
    pop_size = 100
    generations = 50
    num_runs = 50

    config_file = 'novelty_neat/configs/tiling_generate_12_1_balanced_pop10'
    print("Doing now")
    results_directory = f'../results/{name}/{game}/{method}/{date}/{pop_size}/{generations}'
    maze_game = MazeGame(MazeLevel.random())
    level_generators = {
        0: GenerateMazeLevelsUsingTiling(game=maze_game, number_of_random_variables=4,
                                         should_add_coords=False,
                                         do_padding_randomly=False,
                                         should_start_with_full_level=False,
                                         random_perturb_size=0.1565),
        1: GenerateMazeLevelsUsingTiling(game=maze_game, number_of_random_variables=4,
                                         should_add_coords=False,
                                         do_padding_randomly=False,
                                         should_start_with_full_level=False,
                                         random_perturb_size=0.1565,
                                         reverse=0),
    }

    def get_overall_fitness(which) -> NeatFitnessFunction:
        num_levels = 24
        num_levels_other = num_levels
        K = 5
        distance_func = visual_diversity_only_reachable
        max_dist = 196
        return CombinedFitness([
            NoveltyMetric(level_generators[which], distance_func, max_dist=max_dist, number_of_levels=num_levels,
                          number_of_neighbours=K, lambd=1, archive_mode=NoveltyArchive.RANDOM,
                          should_use_all_pairs=False),
            SolvabilityFitness(num_levels_other, level_generators[which]),
            NoveltyIntraGenerator(num_levels, level_generators[which], distance_func, max_dist=max_dist,
                                  number_of_neighbours=min(10, num_levels - 1)),
            PathLengthFitness(num_levels, level_generators[which], should_reward_larger_levels_more=True)
        ], [0.4114, 0.5051, 0.4214, 2],
            number_of_levels_to_generate=num_levels_other, level_gen=level_generators[which], mode='add')

    def get_neat_config():
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_file)

    def get_pop(which):
        level = MazeLevel.random()
        game = MazeGame(level)
        fitness = get_overall_fitness(which)
        return NoveltyNeatPCG(game, level, level_generator=level_generators[which], fitness_calculator=fitness, neat_config=get_neat_config(),
                              num_generations=generations)

    pop1 = get_pop(0)
    pop2 = get_pop(1)
    config = get_neat_config()
    def fitness(pop, genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config):
        nets = []
        ans = {}
        for genome_id, genome in genomes:
            nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        all_fitnesses = pop.fitness_calculator(nets)
        print(all_fitnesses)
        for fit, (id, genome) in zip(all_fitnesses, genomes):
            genome.fitness = fit
            ans[id] = fit
        return ans
    
    five_percent = ceil(pop_size / 20)

    @ray.remote
    def train_pop(pop):
        pop.train(NoneLogger())
        return pop


    def _get_values(pop, pops, config, name):
        fits = fitness(pop, pops.items(), config)
        list_fits = list(fits.values())
        print(f"Pop {name}, min = {np.min(list_fits)}, max = {np.max(list_fits)}, mean = {np.mean(list_fits)} ")
        fit_indices = sorted(map(tuple, map(reversed, fits.items())), reverse=True)
        fit_indices = [(id, pops[id]) for (_, id) in fit_indices]
    
        top_5 = fit_indices[:five_percent]
        # remove bottom 5
        bad_5 = fit_indices[-five_percent:]
        fit_indices = fit_indices[:-five_percent]

        return fit_indices, top_5, bad_5;

    def update_fitnesses(fit_indices, bad_this, top_other):
        fit_indices.extend([
            (id_bad, fit) for ((id_bad, bad_fit), (id_og, fit)) in zip(bad_this, top_other)
        ])
        return fit_indices

    def swap_populations(pop1, pop2):

        # then we exchange the populations...
        pops1 = pop1.pop.population
        pops2 = pop2.pop.population

        fit_indices1, top_5_1, bad_5_1 = _get_values(pop1, pops1, config, 'pop1')
        fit_indices2, top_5_2, bad_5_2 = _get_values(pop2, pops2, config, 'pop2')

        # Now, add in top51 to fit_indices2
        fit_indices2 = update_fitnesses(fit_indices2, bad_5_2, top_5_1)
        
        
        fit_indices1 = update_fitnesses(fit_indices1, bad_5_1, top_5_2)

        # Now, make pop again from the things
        pop1.pop.population = {
            id: genome for id, genome in fit_indices1
        }


        pop2.pop.population = {
            id: genome for id, genome in fit_indices2
        }
    

    for i in range(num_runs):
        print(f"{i+1}/{num_runs}")

        vals = [train_pop.remote(p) for p in [pop1, pop2]]

        new_pops = ray.get(vals)
        pop1 = new_pops[0]
        pop2 = new_pops[1]
        swap_populations(pop1, pop2)

        if len(pop1.fitness_calculator.fitnesses[0].previously_novel_individuals) >= 100:
            pop1.fitness_calculator.fitnesses[0].previously_novel_individuals = []
        
        if len(pop2.fitness_calculator.fitnesses[0].previously_novel_individuals) >= 100:
            pop2.fitness_calculator.fitnesses[0].previously_novel_individuals = []
    

    # generate some levels
    levels = {
        'pop1': [pop1.generate_level() for _ in range(100)],
        'pop2': [pop2.generate_level() for _ in range(100)],
        }
    dirname = f'../results/maze/island/{get_date()}'
    os.makedirs(dirname, exist_ok=True)
    file = f'{dirname}/data.p'
    with open(file, 'wb+') as f:
        dic = {
            'pop1': pop1,
            'pop2': pop2,
            'gens': num_runs * generations,
            'levels': levels
        }
        pickle.dump(dic, f)


def analyse():
    name = 'TODO_ADD_PATH'
    with open(name, 'rb') as f:
        dic = pickle.load(f)
        ls = dic['levels']
        ls1 = ls['pop1']
        ls2 = ls['pop2']
    
    fig, axs = plt.subplots(2, 5, figsize=(20, 20 * 2/3))
    lengths1 = [
        path_length(l) for l in ls1
    ]
    lengths2 = [
        path_length(l) for l in ls2
    ]

    for i in range(len(axs[0])):
        axs[0][i].imshow(1 - ls1[i].map, cmap='gray', vmin=0, vmax=1)
        axs[1][i].imshow(1 - ls2[i].map, cmap='gray', vmin=0, vmax=1)
        
        axs[0][i].set_title(f"Path Length = {lengths1[i]}")
        axs[1][i].set_title(f"Path Length = {lengths2[i]}")
    print(f"Average length: pop1 = {np.mean(lengths1)}, pop={np.mean(lengths2)}")
    plt.show()
    print(dic['gens'])


if __name__ == '__main__':
    run_experiments_test()
    # analyse()
