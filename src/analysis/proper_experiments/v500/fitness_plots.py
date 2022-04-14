from cmath import exp
from functools import partial
import os
from pprint import pprint
import threading
from matplotlib import pyplot as plt
import numpy as np
import wandb
from baselines.ga.genetic_algorithm_pcg import GeneticAlgorithmIndividual, GeneticAlgorithmIndividualMaze, GeneticAlgorithmPCG
from common.types import Verbosity
from common.utils import get_date, load_compressed_pickle, mysavefig
from experiments.experiment import Experiment
from experiments.config import Config
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarEditDistanceDiversityMetric, AStarSolvabilityMetric
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
from novelty_neat.novelty.distance_functions.distance import dist_jensen_shannon_compare_probabilities, euclidean_distance, image_hash_distance_average, image_hash_distance_perceptual, image_hash_distance_perceptual_simple, jensen_shannon_compare_trajectories_distance, jensen_shannon_distance, visual_diversity, visual_diversity_normalised, visual_diversity_only_reachable, image_hash_distance_wavelet, dist_compare_shortest_paths, rolling_window_comparison_what_you_see_from_normal_default, rolling_window_comparison_what_you_see_from_normal_default_TRAJ
import neat
import cProfile
import pstats
import ray
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

""" This saves the fitness values for mario and maze.
"""

def v551_maze_save_fitness_and_models(
    distance_func = visual_diversity_only_reachable, max_dist = 196
):
    ray.init()
    name = 'experiment_551_a'
    game = 'Maze'
    method = 'NEAT'
    date = get_date()
    pop_size = 50
    generations = 200

    config_file = f'runs/proper_experiments/v100_maze/config/tiling_generate_12_1_balanced_pop{pop_size}'
    results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/'
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
        return CombinedFitness([
                                    NoveltyMetric(level_generator, distance_func, max_dist=max_dist, number_of_levels=num_levels, 
                                        number_of_neighbours=K, lambd=0, archive_mode=NoveltyArchive.RANDOM,
                                        should_use_all_pairs=False),
                                    SolvabilityFitness(num_levels_other, level_generator),
                                    NoveltyIntraGenerator(num_levels, level_generator, distance_func, max_dist=max_dist, 
                                        number_of_neighbours=min(10, num_levels - 1))
                                    ], [1, 0.5051, 1],
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
        parent = AStarDiversityAndDifficultyMetric(proper_game, number_of_times_to_do_evaluation=5)
        experiment = Experiment(config, get_pop, [
            CompressionDistanceMetric(proper_game), LinearityMetric(
                proper_game), LeniencyMetric(proper_game),
            AStarSolvabilityMetric(proper_game, parent),
            AveragePairWiseDistanceMetric(proper_game),
            AveragePairWiseDistanceMetricOnlyPlayable(proper_game),
            AStarDiversityMetric(proper_game, parent),
            AStarDifficultyMetric(proper_game, parent),
            AStarEditDistanceDiversityMetric(proper_game, parent),
        ], log_to_wandb=False, log_to_dict=True, verbose=Verbosity.PROGRESS if seed == 0 else Verbosity.NONE)

        experiment.do_all()
        wandb.finish()

        try:
            print('num novel', len(experiment.method.fitness_calculator.fitnesses[0].previously_novel_individuals))
        except Exception as e:
            print("Length failed with msg", e)
        return f"Completed with seed = {seed}"
    
    futures = [single_func.remote(i) for i in range(5)]
    print(ray.get(futures))


def _mario_main(date, _index=-1, pred_size = 1, one_hot = False, generations=50, lambd=1, pop_size=50, num_levels=6, solv_weight=1, name='204b', use_intra_novelty=True,
distance_func=visual_diversity, max_dist=14*114):
    """
        This tries to run NEAT, single population with 5 seeds and proper exp.
        In effect copied from v10
    """
    name = f'{name}'
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
    results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{str(distance_func)}_{str(max_dist)}'
    mario_game = MarioGame(MarioLevel())
    level_generator = GenerateGeneralLevelUsingTiling(mario_game, 1, 4, False, 0, 
                                                      predict_size=pred_size, 
                                                      reversed_direction = 0, 
                                                      use_one_hot_encoding=one_hot)

    def get_overall_fitness() -> NeatFitnessFunction:
        num_levels_other = num_levels
        K = 15
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
        ], log_to_wandb=False, log_to_dict=True, verbose=Verbosity.PROGRESS if seed == 0 else Verbosity.NONE)

        experiment.do_all()
        wandb.finish()

        return f"Completed with seed = {seed}"
    
    futures = [single_func.remote(i) for i in range(5)]
    return futures


def v_552_a_mario_funcs():
    ray.init()
    ray.get(_mario_main(
        pred_size=1,
        one_hot=True,
        generations=150,
        lambd=0,
        pop_size=100,
        num_levels=6,
        solv_weight=2,
        date=get_date(),
        name='552_a',
        distance_func=visual_diversity_normalised,
        max_dist=1
    ))

def analyse(which='Maze'):
    def make_clean(n):
        return {
            'SolvabilityFitness': 'Solvability',
            'MarioSolvabilityFitness': 'Solvability',
            'Total': 'Total Fitness',
            'NoveltyMetric': 'Novelty',
            'NoveltyIntraGenerator': 'Intra-Novelty',
            }[n]
    import seaborn as sns
    sns.set_theme()
    dirs = f'./results/all_models/pcgnn/{which}Game/logs/*.pbz2'
    all = None
    for SS, log_file in enumerate(glob.glob(dirs)):
        result = load_compressed_pickle(log_file)
        LL = len(result) // 2
        components = result[0:len(result)-1:2]
        overall = result[1:len(result)-1:2]
        assert len(components) == LL
        assert components[0] == result[0]
        
        assert len(overall) == LL
        assert overall[0] == result[1]
        
        things = {
            f: {
                'min': [], 'max': [], 'mean': []
            } for f in ['Total'] + list(components[0]['CombinedFitness'].keys())
        }
        if all is None:
            all = {
            f: {
                'min': [[] for i in range(5)], 'max': [[] for i in range(5)], 'mean': [[] for i in range(5)]
            } for f in ['Total'] + list(components[0]['CombinedFitness'].keys())
        }
        for i in range(len(components)):
            for k, v in components[i]['CombinedFitness'].items():
                for p in things[k].keys():
                    things[k][p].append(v[p])
                    all[k][p][SS].append(v[p])
            for p in things['Total'].keys():
                things['Total'][p].append(overall[i][p + "_fitness"])
                all['Total'][p][SS].append(overall[i][p + "_fitness"])
        
    fig, axs = plt.subplots(1, len(things), figsize=(20, 5))
    axs = axs.ravel()
    for (k, v), ax in zip(all.items(), axs):
        for kk, vv in v.items():
            MM = np.mean(vv, axis=0)
            Std = np.std(vv, axis=0)
            XX=  np.arange(len(MM))
            ax.plot(XX, MM, label=kk)
            ax.fill_between(XX, MM - Std, MM+Std, alpha=0.3)
        ax.set_title(make_clean(k), fontsize=24)
        ax.set_xlabel("Generation Number", fontsize=16)
    axs[0].set_ylabel("Fitness Value", fontsize=16)
        
    axs[0].legend(fontsize=16, loc='lower right')
    plt.tight_layout()
    mysavefig(f'./results/all_models/pcgnn/{which}Game/fitness_plots_{which}.png', bbox_inches='tight', pad_inches=0)
    
        
        
    

def main():
    DO_MAZE = int(sys.argv[-1])
    if DO_MAZE == 1:
        v551_maze_save_fitness_and_models(visual_diversity_only_reachable, 196)
    elif DO_MAZE == 0:
        v_552_a_mario_funcs()
    else:
        analyse('Maze')
        analyse('Mario')
    
if __name__ == '__main__':
    main()   