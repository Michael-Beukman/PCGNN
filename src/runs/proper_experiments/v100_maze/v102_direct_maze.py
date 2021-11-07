import cProfile
import glob
from math import ceil
import os
from cmath import exp
from pprint import pprint
import sys
from time import sleep
import neat
import numpy as np
import ray
from torch import futures
import wandb
from baselines.ga.direct_ga_fitness import DirectFitness
from baselines.ga.general_ga_pcg import GeneralGAPCG, _GeneralGAIndividualMaze
from common.utils import get_date
from experiments.config import Config
from experiments.experiment import Experiment
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.average_pairwise_distance import (
    AveragePairWiseDistanceMetric, AveragePairWiseDistanceMetricOnlyPlayable)
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.horn.linearity import LinearityMetric
from metrics.rl.tabular.rl_agent_metric import RLAgentMetric
from metrics.rl.tabular.rl_difficulty_metric import RLDifficultyMetric
from metrics.solvability import SolvabilityMetric
from novelty_neat.fitness.entropy import EntropyFitness
from novelty_neat.fitness.fitness import CombinedFitness
from novelty_neat.maze.neat_maze_fitness import PartialSolvabilityFitness, SolvabilityFitness
from novelty_neat.novelty.distance_functions.distance import (
    euclidean_distance, image_hash_distance_perceptual_simple, visual_diversity)
from novelty_neat.novelty.novelty_metric import NoveltyArchive, NoveltyMetric, NoveltyMetricDirectGA
from novelty_neat.novelty_neat import NoveltyNeatPCG
import fire

os.environ['WANDB_SILENT'] = 'True'


def experiment_102_a_rerun():
    """
        This is experiment 102 a rerun, which basically just reruns the following, as they were good.
        It reruns them to get an accurate level generation time estimate.
    """
    # One date
    date = get_date()
    ray.init(num_cpus=14)

    @ray.remote
    def single_func(seed, index, name, game, method, results_directory, args, date, get_pop):
        # A single function to run one seed and one param combination.
        config = Config(
            name=name,
            game=game,
            method=method,
            seed=seed,
            results_directory=os.path.join(results_directory, str(seed)),
            method_parameters=args,
            date=date,
            project='NoveltyNeatPCG-Experiment102'
        )
        print("Date = ", config.date, config.results_directory,
              config.hash(seed=False))
        proper_game = MazeGame(MazeLevel())
        experiment = Experiment(config, get_pop, [
            CompressionDistanceMetric(proper_game),
            LinearityMetric(proper_game),
            LeniencyMetric(proper_game),
            SolvabilityMetric(proper_game),
            AveragePairWiseDistanceMetric(proper_game),
            AveragePairWiseDistanceMetricOnlyPlayable(proper_game),
            RLAgentMetric(proper_game),
            RLDifficultyMetric(proper_game, N=100)
        ], log_to_wandb=True)
        experiment.do_all()
        wandb.finish()

        return f"Completed index = {index} with seed = {seed}"

    def one_run(index, use_solv: bool, desired_entropy: float, pop_size: int, num_gens: int, use_novelty: bool):
        name = 'experiment_102_aaa_rerun_only_best'
        game = 'Maze'
        method = 'DirectGA'
        generations = num_gens

        print(
            f"Doing now, with params = Solvability={use_solv}, Desired Entropy = {desired_entropy:>3}, Pop Size = {pop_size:>3}, Generations = {num_gens:>3}, Novelty = {use_novelty}")
        results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{pop_size}/{generations}/{desired_entropy}/{use_solv}/{use_novelty}'
        maze_game = MazeGame(MazeLevel.random())

        def get_overall_fitness() -> DirectFitness:
            num_levels = 1
            level_generator = None
            fitness = [EntropyFitness(
                num_levels, level_generator, desired_entropy=desired_entropy)]

            if use_novelty:
                fitness.append(NoveltyMetricDirectGA(image_hash_distance_perceptual_simple, max_dist=1, number_of_levels=1,
                                                     number_of_neighbours=6 if pop_size <= 15 else 15, lambd=1, archive_mode=NoveltyArchive.RANDOM))

            if use_solv:
                fitness.append(PartialSolvabilityFitness(
                    num_levels, level_generator))

            weights = np.ones(len(fitness))

            return DirectFitness(CombinedFitness(fitness, weights, number_of_levels_to_generate=num_levels, level_gen=level_generator))
        args = {
            'population_size': pop_size,
            'number_of_generations': generations,
            'fitness': get_overall_fitness().params(),
            'desired_entropy': desired_entropy,
            'use_solvability': use_solv,
            'use_novelty': use_novelty,
            'level_gen': 'DirectGA'
        }

        def get_pop():
            level = MazeLevel.random()
            game = MazeGame(level)
            fitness = get_overall_fitness()
            return GeneralGAPCG(game, level, lambda l: _GeneralGAIndividualMaze(l), fitness, pop_size, generations)

        futures = [single_func.remote(
            seed=i, index=index, name=name, game=game, method=method, results_directory=results_directory, args=args, date=date, get_pop=get_pop
        ) for i in range(5)]
        # We return the ray futures, and then only get them later, so that ray can run more than 5 at a time.
        return futures

    # The grid search thing

    confs = [
        {
            'use_solv': True,
            'desired_entropy': 1,
            'pop_size': 100,
            'num_gens': 100,
            'use_novelty': False
        }
    ]

    counter = 0
    all_futures = []
    for c in confs:
        counter += 1
        all_futures += one_run(counter, **c)
    print(
        f"At end, we have {counter} runs to do. Running experiment_102_a_rerun")
    ray.get(all_futures)
    print("Done")


def experiment_102_f():
    """
    2021/10/03
    This runs the best visual diversity one from 102e on batch to get accurate timeing scores.
    """

    # One date
    date = get_date()
    ray.init(num_cpus=12)

    @ray.remote
    def single_func(seed, index, name, game, method, results_directory, args, date, get_pop):
        # A single function to run one seed and one param combination.
        config = Config(
            name=name,
            game=game,
            method=method,
            seed=seed,
            results_directory=os.path.join(results_directory, str(seed)),
            method_parameters=args,
            date=date,
            project='NoveltyNeatPCG-Experiment102'
        )
        print("Date = ", config.date, config.results_directory,
              config.hash(seed=False))
        proper_game = MazeGame(MazeLevel())
        experiment = Experiment(config, get_pop, [
            CompressionDistanceMetric(proper_game),
            LinearityMetric(proper_game),
            LeniencyMetric(proper_game),
            SolvabilityMetric(proper_game),
            AveragePairWiseDistanceMetric(proper_game),
            AveragePairWiseDistanceMetricOnlyPlayable(proper_game),
            RLAgentMetric(proper_game),
            RLDifficultyMetric(proper_game, N=100)
        ], log_to_wandb=True)
        experiment.do_all()
        wandb.finish()

        return f"Completed index = {index} with seed = {seed}"

    def one_run(index, use_solv: bool, desired_entropy: float, pop_size: int, num_gens: int, use_novelty: bool):
        name = 'experiment_102_f_visual_diversity_rerun_batch'
        game = 'Maze'
        method = 'DirectGA'
        generations = num_gens

        print(
            f"Doing now, with params = Solvability={use_solv}, Desired Entropy = {desired_entropy:>3}, Pop Size = {pop_size:>3}, Generations = {num_gens:>3}, Novelty = {use_novelty}")
        results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{pop_size}/{generations}/{desired_entropy}/{use_solv}/{use_novelty}'
        maze_game = MazeGame(MazeLevel.random())

        def get_overall_fitness() -> DirectFitness:
            num_levels = 1
            level_generator = None
            fitness = [EntropyFitness(
                num_levels, level_generator, desired_entropy=desired_entropy)]

            if use_novelty:
                fitness.append(NoveltyMetricDirectGA(visual_diversity, max_dist=196, number_of_levels=1,
                                                     number_of_neighbours=6 if pop_size <= 15 else 15, lambd=1, archive_mode=NoveltyArchive.RANDOM))

            if use_solv:
                fitness.append(PartialSolvabilityFitness(
                    num_levels, level_generator))

            weights = np.ones(len(fitness))

            return DirectFitness(CombinedFitness(fitness, weights, number_of_levels_to_generate=num_levels, level_gen=level_generator))
        args = {
            'population_size': pop_size,
            'number_of_generations': generations,
            'fitness': get_overall_fitness().params(),
            'desired_entropy': desired_entropy,
            'use_solvability': use_solv,
            'use_novelty': use_novelty,
            'level_gen': 'DirectGA'
        }

        def get_pop():
            level = MazeLevel.random()
            game = MazeGame(level)
            fitness = get_overall_fitness()
            return GeneralGAPCG(game, level, lambda l: _GeneralGAIndividualMaze(l), fitness, pop_size, generations)

        futures = [single_func.remote(
            seed=i, index=index, name=name, game=game, method=method, results_directory=results_directory, args=args, date=date, get_pop=get_pop
        ) for i in range(5)]
        # We return the ray futures, and then only get them later, so that ray can run more than 5 at a time.
        return futures

    counter = 0
    all_futures = []
    conf = {
        'use_solv': True,
        'desired_entropy': 0,
        'pop_size': 50,
        'num_gens': 100,
        'use_novelty': True,
    }
    counter += 1
    all_futures += one_run(counter, **conf)
    print(f"At end, we have {counter} runs to do. Running experiment_102_e")
    ray.get(all_futures)
    print("Done")


if __name__ == '__main__':
    I = int(sys.argv[-1])
    if I == 0:
        print("RUNNING 102 aaa - DirectGA+ for Maze")
        experiment_102_a_rerun()
    elif I == 1:
        print("RUNNING 102f - DirectGA (Novelty) for Maze")
        experiment_102_f()
    else:
        assert False, f"Unsupported option {I}"
