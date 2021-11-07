from cmath import exp
from functools import partial
import os
from pprint import pprint
import threading
import wandb
from baselines.ga.genetic_algorithm_pcg import GeneticAlgorithmIndividual, GeneticAlgorithmIndividualMaze, GeneticAlgorithmPCG
from common.types import Verbosity
from common.utils import get_date
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
from novelty_neat.novelty.distance_functions.distance import dist_jensen_shannon_compare_probabilities, euclidean_distance, image_hash_distance_average, image_hash_distance_perceptual, image_hash_distance_perceptual_simple, jensen_shannon_compare_trajectories_distance, jensen_shannon_distance, visual_diversity, visual_diversity_only_reachable, image_hash_distance_wavelet, dist_compare_shortest_paths
import neat
import cProfile
import pstats
import ray

os.environ['WANDB_SILENT'] = 'True'


def experiment_105a():
    ray.init()
    name = 'experiment_105_a'
    game = 'Maze'
    method = 'NEAT'
    date = get_date()
    pop_size = 50
    generations = 200

    config_file = f'runs/proper_experiments/v100_maze/config/tiling_generate_12_1_balanced_pop{pop_size}'
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

if __name__ == '__main__':
    experiment_105a()