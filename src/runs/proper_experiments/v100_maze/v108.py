# This does the following:
# Run NEAT & DirectGA for many generations, and compare how long the generation takes.

from cmath import exp
from functools import partial
import os
from pprint import pprint
import sys
import threading
import numpy as np
import wandb
from baselines.ga.direct_ga_fitness import DirectFitness
from baselines.ga.general_ga_pcg import _GeneralGAIndividualMaze, GeneralGAPCG
from baselines.ga.genetic_algorithm_pcg import GeneticAlgorithmIndividual, GeneticAlgorithmIndividualMaze, GeneticAlgorithmPCG
from common.types import Verbosity
from common.utils import get_date
from experiments.experiment import Experiment
from experiments.config import Config
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarEditDistanceDiversityMetric
from metrics.average_pairwise_distance import AveragePairWiseDistanceMetric, AveragePairWiseDistanceMetricOnlyPlayable
from metrics.combination_metrics import RLAndSolvabilityMetric
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.horn.linearity import LinearityMetric
from metrics.rl.tabular.rl_agent_metric import RLAgentMetric
from metrics.rl.tabular.rl_difficulty_metric import RLDifficultyMetric
from metrics.solvability import SolvabilityMetric
import os
from novelty_neat.fitness.entropy import EntropyFitness
from novelty_neat.fitness.fitness import CombinedFitness, NeatFitnessFunction
from novelty_neat.maze.neat_maze_fitness import PartialSolvabilityFitness, SolvabilityFitness
from novelty_neat.maze.neat_maze_level_generation import GenerateMazeLevelsUsingMoreContext, GenerateMazeLevelsUsingTiling, GenerateMazeLevelsUsingCPPNCoordinates, GenerateMazeLevelsUsingTilingVariableTileSize
from novelty_neat.novelty_neat import NoveltyNeatPCG
from novelty_neat.novelty.novelty_metric import NoveltyArchive, NoveltyIntraGenerator, NoveltyMetric
import neat
from novelty_neat.novelty.distance_functions.distance import visual_diversity_only_reachable
import ray


def v108_a_neat(generations: int):
    # Does with NEAT
    ray.init(num_cpus=5)
    name = 'experiment_108_a'
    game = 'Maze'
    method = 'NEAT'
    date = get_date()
    pop_size = 50

    config_file = 'runs/proper_experiments/v100_maze/config/tiling_generate_12_1_balanced_pop100'
    print("Doing now")
    results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{pop_size}/{generations}'
    maze_game = MazeGame(MazeLevel.random())
    level_generator = GenerateMazeLevelsUsingTiling(game=maze_game, number_of_random_variables=4, 
                        should_add_coords=False,
                        do_padding_randomly=False,
                        should_start_with_full_level=False, 
                        random_perturb_size=0.1565)
    def get_overall_fitness() -> NeatFitnessFunction:
        num_levels = 24
        num_levels_other = num_levels
        K = 15
        distance_func = visual_diversity_only_reachable; max_dist = 196
        return CombinedFitness([
                                    NoveltyMetric(level_generator, distance_func, max_dist=max_dist, number_of_levels=num_levels, 
                                        number_of_neighbours=K, lambd=2, archive_mode=NoveltyArchive.RANDOM,
                                        should_use_all_pairs=False),
                                    SolvabilityFitness(num_levels_other, level_generator),
                                    NoveltyIntraGenerator(num_levels, level_generator, distance_func, max_dist=max_dist, 
                                        number_of_neighbours=min(10, num_levels - 1))
                                    ], [0.4114, 0.5051, 0.4214],
                                  number_of_levels_to_generate=num_levels_other, level_gen=level_generator, mode='add')
    args = {
        'population_size': pop_size,
        'number_of_generations': generations,
        'fitness': get_overall_fitness().params(),
        'level_gen': 'tiling',
        'config_filename': config_file
    }
    print(args)
    def get_neat_config():
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_file)
    
    def get_pop():
        level = MazeLevel.random()
        game = MazeGame(level)
        fitness = get_overall_fitness()
        return NoveltyNeatPCG(game, level, level_generator=level_generator, fitness_calculator=fitness, neat_config=get_neat_config(),
                              num_generations=generations)

    @ray.remote
    def single_func(seed):
        config = Config(
            name=name,
            game=game,
            method=method,
            seed=seed,
            results_directory=os.path.join(results_directory, str(seed)),
            method_parameters=args,
            date=date
        )
        print("Date = ", config.date, config.results_directory, config.hash(seed=False))
        proper_game = MazeGame(MazeLevel())
        experiment = Experiment(config, get_pop, [
            CompressionDistanceMetric(proper_game), LinearityMetric(
                proper_game), LeniencyMetric(proper_game),
            SolvabilityMetric(proper_game),
            AveragePairWiseDistanceMetric(proper_game),
            AveragePairWiseDistanceMetricOnlyPlayable(proper_game),
        ], log_to_wandb=True, verbose=Verbosity.PROGRESS if seed == 0 else Verbosity.NONE)

        experiment.do_all()
        wandb.finish()

        try:
            print('num novel', len(experiment.method.fitness_calculator.fitnesses[0].previously_novel_individuals))
        except Exception as e:
            print("Length failed with msg", e)
        return f"Completed with seed = {seed}"
    
    futures = [single_func.remote(i) for i in range(5)]
    print(ray.get(futures))


def v108_b_directga(number_of_generations):
    """
    """
    # One date
    date = get_date()
    ray.init(num_cpus=5)

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
            project='NoveltyNeatPCG'
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
        ], log_to_wandb=True)
        experiment.do_all()
        wandb.finish()

        return f"Completed index = {index} with seed = {seed}"

    def one_run(index, use_solv: bool, desired_entropy: float, pop_size: int, num_gens: int, use_novelty: bool):
        name = 'experiment_108'
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
            'pop_size': 50,
            'num_gens': number_of_generations,
            'use_novelty': False
        }
    ]

    counter = 0
    all_futures = []
    for c in confs:
        counter += 1
        all_futures += one_run(counter, **c)
    print(
        f"At end, we have {counter} runs to do. Running experiment_108b")
    ray.get(all_futures)
    print("Done")

def make_slurms():
    s = """#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J 108_{WHICH}_{GENS}
#SBATCH -o /home/NAME/PATH_TO_ROOT/src/logs/pipelines/108_{WHICH}_{GENS}.%N.%j.out

source ~/.bashrc
cd /home/NAME/PATH_TO_ROOT/src
conda activate noveltyneatpcg
echo "Doing V108_{WHICH}_{GENS}"
./run.sh runs/proper_experiments/v100_maze/v108.py {WHICH} {GENS}
"""
    gens = [10, 20, 40, 80, 160, 320]
    for which in [0, 1]:
        for g in gens:
            formatted = s.format(WHICH=which, GENS=g)
            with open(f'pipelines/v100/v108/v108_{which}_{g}.batch', 'w+') as f:
                f.write(formatted)

if __name__ == '__main__':
    which, num_gens = sys.argv[1:]
    which = int(which)
    num_gens = int(num_gens)
    if which == 0:
        print(f"DOING NEAT with gens = {num_gens}")
        v108_a_neat(num_gens)
    elif which == 1:
        print(f"DOING DirectGA with gens = {num_gens}")
        v108_b_directga(num_gens)
    else:
        assert False, "BAD run id"