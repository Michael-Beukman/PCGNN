import os
from pprint import pprint

import neat
import ray
import wandb

from common.types import Verbosity
from common.utils import get_date
from experiments.config import Config
from experiments.experiment import Experiment
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric
from metrics.average_pairwise_distance import (
    AveragePairWiseDistanceMetric, AveragePairWiseDistanceMetricOnlyPlayable)
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.horn.linearity import LinearityMetric
from metrics.solvability import SolvabilityMetric
from novelty_neat.fitness.fitness import CombinedFitness, NeatFitnessFunction
from novelty_neat.fitness.mario.mario_fitness import (MarioSolvabilityFitness)
from novelty_neat.general.neat_generate_general_level import \
    GenerateGeneralLevelUsingTiling
from novelty_neat.island.novelty_neat_island_model import \
    NoveltyNeatIslandModel
from novelty_neat.maze.neat_maze_level_generation import (
    GenerateMazeLevelsUsingCPPNCoordinates, GenerateMazeLevelsUsingMoreContext,
    GenerateMazeLevelsUsingTiling,
    GenerateMazeLevelsUsingTilingVariableTileSize)
from novelty_neat.novelty.distance_functions.distance import (
    dist_compare_shortest_paths, dist_jensen_shannon_compare_probabilities,
    euclidean_distance, image_hash_distance_average,
    image_hash_distance_perceptual, image_hash_distance_perceptual_simple,
    image_hash_distance_wavelet, jensen_shannon_compare_trajectories_distance,
    jensen_shannon_distance, trajectory_sample_distance, visual_diversity,
    visual_diversity_only_reachable)
from novelty_neat.novelty.novelty_metric import (NoveltyArchive,
                                                 NoveltyIntraGenerator,
                                                 NoveltyMetric)
from novelty_neat.novelty_neat import NoveltyNeatPCG


def main(date, pred_size = 1, one_hot = False, num_things=5):
    """
    This tries to run the island neat model, in a proper experiment way.
    """
    name = 'experiment_203'
    game = 'Mario'
    method = 'NeatNoveltyIsland'
    message = f'Island Mario, experiment 203, pred size={pred_size}, one hot = {one_hot}, num gens and num steps = {num_things}'
    pop_size = 50
    generations = num_things
    number_of_migration_steps = num_things
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


    print("Doing now")
    results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{pop_size}/{generations}/{one_hot}/{pred_size}/{num_things}'
    mario_game = MarioGame(MarioLevel())
    level_generators = {
        0: GenerateGeneralLevelUsingTiling(mario_game, 1, 4, False, predict_size=pred_size, reversed_direction = 0, random_perturb_size=0.1, use_one_hot_encoding=one_hot),
        1: GenerateGeneralLevelUsingTiling(mario_game, 1, 4, False, predict_size=pred_size, reversed_direction = 1, random_perturb_size=0.1, use_one_hot_encoding=one_hot)
    }

    def get_overall_fitness(which) -> NeatFitnessFunction:
        num_levels = 6
        num_levels_other = 6
        K = 15
        distance_func = visual_diversity; max_dist = 14 * 114
        return CombinedFitness([
                                    NoveltyMetric(level_generators[which], 
                                        distance_func, 
                                        max_dist=max_dist, number_of_levels=num_levels, 
                                        number_of_neighbours=K, lambd=0, archive_mode=NoveltyArchive.RANDOM,
                                        should_use_all_pairs=False,
                                        distance_mode='distance'),
                                    MarioSolvabilityFitness(num_levels, level_generators[which]),
                                    NoveltyIntraGenerator(num_levels, level_generators[which], distance_func, max_dist=max_dist, number_of_neighbours=2),
                                    ], [1, 1, 1],
                                  number_of_levels_to_generate=num_levels_other, level_gen=level_generators[which], mode='add')
    def get_neat_config():
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_file)

    def get_pop():
        level = MarioLevel()
        game = MarioGame(level)
        all_models = []
        for which in level_generators:
            fitness = get_overall_fitness(which)
            all_models.append(NoveltyNeatPCG(game, level, level_generator=level_generators[which], fitness_calculator=fitness, neat_config=get_neat_config(),
                                num_generations=generations))
        return NoveltyNeatIslandModel(game, level, all_models, number_of_migration_steps=number_of_migration_steps)

    args = {
        'population_size': pop_size,
        'number_of_generations': generations,
        'number_of_migration_steps': number_of_migration_steps,
        'fitnesses': 
            {
                i: get_overall_fitness(i).params() for i in level_generators.keys()
            },
        'level_gen': 'tiling',
        'config_filename': config_file,
        'message': message,

        'pred_size': pred_size,
        'one_hot': one_hot
    }
    print(args)
    def get_neat_config():
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_file)
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
        parent = AStarDiversityAndDifficultyMetric(proper_game)

        experiment = Experiment(config, get_pop, [
            CompressionDistanceMetric(proper_game), LinearityMetric(
                proper_game), LeniencyMetric(proper_game),
            SolvabilityMetric(proper_game),
            AveragePairWiseDistanceMetric(proper_game),
            AveragePairWiseDistanceMetricOnlyPlayable(proper_game),
            AStarDifficultyMetric(proper_game, parent),
            AStarDiversityMetric(proper_game, parent),
        ], log_to_wandb=True, verbose=Verbosity.PROGRESS if seed == 0 else Verbosity.NONE)

        experiment.do_all()
        wandb.finish()
        
        return f"Completed with seed = {seed}"
    
    futures = [single_func.remote(i) for i in range(5)]
    print(ray.get(futures)) # [0, 1, 4, 9]



if __name__ == '__main__':
    ray.init()
    date = get_date()
    for things in [5, 10]:
        for pred_size in [1, 2]:
            for one_hot in [True, False]:
                print(f"RUNNING NOW Things = {things}, PSize={pred_size} | ONE_HOT = {one_hot}")
                main(date, pred_size, one_hot)
    