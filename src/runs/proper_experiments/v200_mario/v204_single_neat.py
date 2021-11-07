from cmath import exp
from functools import partial
import glob
from math import ceil
import os
from pprint import pprint
import sys
import threading
import wandb
from baselines.ga.genetic_algorithm_pcg import GeneticAlgorithmIndividual, GeneticAlgorithmIndividualMaze, GeneticAlgorithmPCG
from common.types import Verbosity
from common.utils import get_date
from experiments.experiment import Experiment
from experiments.config import Config
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarSolvabilityMetric
from metrics.average_pairwise_distance import AveragePairWiseDistanceMetric, AveragePairWiseDistanceMetricOnlyPlayable
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.horn.linearity import LinearityMetric
from metrics.rl.tabular.rl_agent_metric import RLAgentMetric
from metrics.rl.tabular.rl_difficulty_metric import RLDifficultyMetric
from metrics.solvability import SolvabilityMetric
import os
from novelty_neat.fitness.entropy import EntropyFitness
from novelty_neat.fitness.fitness import CombinedFitness, NeatFitnessFunction
from novelty_neat.fitness.mario.mario_fitness import MarioFeasibilityFitness, MarioNumberEmptyTiles, MarioSolvabilityFitness
from novelty_neat.general.neat_generate_general_level import GenerateGeneralLevelUsingTiling
from novelty_neat.maze.neat_maze_fitness import PartialSolvabilityFitness, SolvabilityFitness
from novelty_neat.maze.neat_maze_level_generation import GenerateMazeLevelsUsingMoreContext, GenerateMazeLevelsUsingTiling, GenerateMazeLevelsUsingCPPNCoordinates, GenerateMazeLevelsUsingTilingVariableTileSize
from novelty_neat.novelty_neat import NoveltyNeatPCG
from novelty_neat.novelty.novelty_metric import NoveltyArchive, NoveltyIntraGenerator, NoveltyMetric
from novelty_neat.novelty.distance_functions.distance import dist_jensen_shannon_compare_probabilities, euclidean_distance, image_hash_distance_average, image_hash_distance_perceptual, image_hash_distance_perceptual_simple, jensen_shannon_compare_trajectories_distance, jensen_shannon_distance, trajectory_sample_distance, visual_diversity, visual_diversity_only_reachable, image_hash_distance_wavelet, dist_compare_shortest_paths
import neat
import cProfile
import pstats
import ray

os.environ['WANDB_SILENT'] = 'True'


def main(date, _index=-1, pred_size = 1, one_hot = False, generations=50, lambd=1, pop_size=50, num_levels=6, solv_weight=1, name='204b', use_intra_novelty=True):
    """
        This tries to run NEAT, single population with 5 seeds and proper exp.
        In effect copied from v10
    """
    name = f'experiment_{name}'
    game = 'Mario'
    method = 'NeatNovelty'
    message = f'Tries to do Mario, single population with neat PredSize = {pred_size}, one hot = {one_hot}, gens = {generations}, date = {date}. Lambda = {lambd}, pop size = {pop_size}. Index = {_index}. Num levels = {num_levels}, solvability weight = {solv_weight}. With Use Intra Novelty = {use_intra_novelty}'

    if one_hot == False:
        if pred_size == 1:
            config_file = './novelty_neat/configs/tiling_mario_12_7_balanced'
        else:
            config_file = './novelty_neat/configs/tiling_mario_20_28_2pred_size'
    else:
        if pred_size == 1:
            config_file = './novelty_neat/configs/tiling_mario_56_7_1pred_size_one_hot'
        else:
            config_file = './novelty_neat/configs/tiling_mario_20_28_2pred_size_one_hot_116'

    if pop_size == 20:
        config_file += "_20_pop_clean"
    if pop_size == 100:
        config_file += "_100_pop_clean"
    if pop_size == 200:
        config_file += "_200_pop_clean"

    print("Doing now")
    results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{pop_size}/{generations}/{one_hot}/{pred_size}/{lambd}/{num_levels}/{solv_weight}/{_index}/{use_intra_novelty}'
    mario_game = MarioGame(MarioLevel())
    level_generator = GenerateGeneralLevelUsingTiling(mario_game, 1, 4, False, 0, 
                                                      predict_size=pred_size, 
                                                      reversed_direction = 0, 
                                                      use_one_hot_encoding=one_hot)

    def get_overall_fitness() -> NeatFitnessFunction:
        num_levels_other = num_levels
        K = 15
        distance_func = visual_diversity; max_dist = 14 * 114
        fitnesses = [
                                    NoveltyMetric(level_generator, 
                                        distance_func, 
                                        max_dist=max_dist, number_of_levels=num_levels, 
                                        number_of_neighbours=K, lambd=lambd, archive_mode=NoveltyArchive.RANDOM,
                                        should_use_all_pairs=False,
                                        distance_mode='distance'),
                                    MarioSolvabilityFitness(num_levels, level_generator),
                                    NoveltyIntraGenerator(num_levels, level_generator, distance_func, max_dist=max_dist, number_of_neighbours=2),
                                    ]
        weights = [1, solv_weight, 1]
        if not use_intra_novelty:
            fitnesses = fitnesses[:-1]
            weights = weights[:-1]
        
        return CombinedFitness(fitnesses, weights,
                                  number_of_levels_to_generate=num_levels_other, level_gen=level_generator, mode='add')
    args = {
        'population_size': pop_size,
        'number_of_generations': generations,
        'fitness': get_overall_fitness().params(),
        'level_gen': 'tiling',
        'config_filename': config_file,
        'message': message,
        'lambd': lambd
    }
    print(args)
    def get_neat_config():
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_file)
    
    def get_pop():
        level = MarioLevel()
        game = MarioGame(level)
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
        proper_game = MarioGame(MarioLevel(), do_enemies=True)
        parent = AStarDiversityAndDifficultyMetric(proper_game, number_of_times_to_do_evaluation=5)

        experiment = Experiment(config, get_pop, [
            CompressionDistanceMetric(proper_game), 
            LeniencyMetric(proper_game),
            AStarSolvabilityMetric(proper_game, parent),
            AStarDifficultyMetric(proper_game, parent),
            AStarDiversityMetric(proper_game, parent),
        ], log_to_wandb=True, verbose=Verbosity.PROGRESS if seed == 0 else Verbosity.NONE)

        experiment.do_all()
        wandb.finish()

        return f"Completed with seed = {seed}"
    
    futures = [single_func.remote(i) for i in range(5)]
    return futures


def main_204e():
    ray.init()
    # This simply runs the best result from above, using  the params of
    # ../results/experiments/experiment_204b/Mario/NeatNovelty/2021-10-07_07-00-00/100/150/True/1/0/6/2/10
    # We rerun this on batch to get accurate training times and such.
    ray.get(main(
        pred_size=1,
        one_hot=True,
        generations=150,
        lambd=0,
        pop_size=100,
        num_levels=6,
        solv_weight=2,
        date=get_date(),
        name='204e'
    ))


if __name__ == '__main__':
    main_204e()
    