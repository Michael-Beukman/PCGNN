import os
import numpy as np
import ray
import wandb
from baselines.ga.direct_ga_fitness import DirectFitness
from baselines.ga.general_ga_pcg import _GeneralGAIndividual, _GeneralGAIndividualMaze, GeneralGAPCG

from common.utils import get_date
from experiments.config import Config
from experiments.experiment import Experiment
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarEditDistanceDiversityMetric, AStarSolvabilityMetric
from metrics.average_pairwise_distance import AveragePairWiseDistanceMetric
from metrics.diversity.simple_diversity import EditDistanceMetric, HammingDistanceMetric
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.horn.linearity import LinearityMetric
from metrics.rl.tabular.rl_agent_metric import RLAgentMetric
from metrics.rl.tabular.rl_difficulty_metric import RLDifficultyMetric
from metrics.solvability import SolvabilityMetric
from novelty_neat.fitness.entropy import EntropyFitness
from novelty_neat.fitness.fitness import CombinedFitness
from novelty_neat.maze.neat_maze_fitness import PartialSolvabilityFitness
from novelty_neat.novelty.distance_functions.distance import image_hash_distance_perceptual_simple
from novelty_neat.novelty.novelty_metric import NoveltyArchive, NoveltyMetricDirectGA


def experiment_107a():
    """
    This is basically the same as experiment 104, but using the normal GA instead of neat to compare.
    """
    date = get_date()
    ray.init(num_cpus=10)

    @ray.remote
    def single_func(seed, index, name, game, method, results_directory, args, date, get_pop, w):
        # A single function to run one seed and one param combination.

        proper_game = MazeGame(MazeLevel(w, w))
        # get the metrics
        N = int(np.ceil(w / 14) ** 2 * 100)
        N = 20 * w
        n_samples = 2 * w
        if w <= 14: 
            N = 100
            n_samples = 30
        if w == 10: 
            N = 50

        args['RL_Difficulty_N'] = N
        args['level_size_width'] = w
        args['RL_Diversity_nsample'] = n_samples
        print(f"N = {N}, N_samples = {n_samples}, proper_game_size = {proper_game.level.map.shape}")
        config = Config(
            name=name,
            game=game,
            method=method,
            seed=seed,
            results_directory=os.path.join(results_directory, str(seed)),
            method_parameters=args,
            date=date,
            project='NoveltyNeatPCG-Experiment107'
        )
        print("Date = ", config.date, config.results_directory,
                config.hash(seed=False))
        
        g = proper_game
        parent = AStarDiversityAndDifficultyMetric(g, number_of_times_to_do_evaluation=5)
        div = AStarDiversityMetric(g, parent)
        diff = AStarDifficultyMetric(g, parent)
        solv = AStarSolvabilityMetric(g, parent)
        edit_distance_div = AStarEditDistanceDiversityMetric(g, parent)

        metrics = [
                solv,
                diff,
                div,
                edit_distance_div,
                LeniencyMetric(g),
                LinearityMetric(g),
                CompressionDistanceMetric(g),
                AveragePairWiseDistanceMetric(g),
                HammingDistanceMetric(g),
                EditDistanceMetric(g),                
            ]


        experiment = Experiment(config, get_pop, metrics, log_to_wandb=True)
        experiment.do_all()
        print(f"Generation time for size = {w} = {experiment.generation_time}")
        wandb.finish()

        return f"Completed index = {index} with seed = {seed}"
    
    def one_run(index, use_solv: bool, desired_entropy: float, pop_size: int, num_gens: int, use_novelty: bool,
                level_size: int):
        name = 'experiment_107_a'
        game = 'Maze'
        method = 'DirectGA'
        generations = num_gens

        print(
            f"Doing now Size = {level_size}, with params = Solvability={use_solv}, Desired Entropy = {desired_entropy:>3}, Pop Size = {pop_size:>3}, Generations = {num_gens:>3}, Novelty = {use_novelty}")
        results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{level_size}/{pop_size}/{generations}/{desired_entropy}/{use_solv}/{use_novelty}'

        def get_overall_fitness() -> DirectFitness:
            num_levels = 1
            level_generator = None
            fitness = [EntropyFitness(
                num_levels, level_generator, desired_entropy=desired_entropy)]

            if use_novelty:
                fitness.append(NoveltyMetricDirectGA(image_hash_distance_perceptual_simple, max_dist=1, number_of_levels=1, 
                    number_of_neighbours= 6 if pop_size <= 15 else 15, lambd=1, archive_mode=NoveltyArchive.RANDOM))

            if use_solv:
                fitness.append(PartialSolvabilityFitness(num_levels, level_generator))

            weights = np.ones(len(fitness))

            return DirectFitness(CombinedFitness(fitness, weights, number_of_levels_to_generate=num_levels, level_gen=level_generator))
        args = {
            'population_size': pop_size,
            'number_of_generations': generations,
            'fitness': get_overall_fitness().params(),
            'desired_entropy': desired_entropy,
            'use_solvability': use_solv,
            'use_novelty': use_novelty,
            'level_gen': 'DirectGA',
            'level_size': level_size
        }

        def get_pop():
            level: MazeLevel = MazeLevel.random(level_size, level_size)
            assert level.map.shape == (level_size, level_size)
            print("LEVEL Map size = ", level.map.shape)
            game = MazeGame(level)
            fitness = get_overall_fitness()
            return GeneralGAPCG(game, level, lambda l: _GeneralGAIndividualMaze(l), fitness, pop_size, generations)

        futures = [single_func.remote(
            seed=i, index=index, name=name, game=game, method=method, results_directory=results_directory, args=args, date=date, get_pop=get_pop, w=level_size
        ) for i in range(5)]
        # We return the ray futures, and then only get them later, so that ray can run more than 5 at a time.
        return futures

    # The grid search thing

    Ws = [10, 14, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # Ws = [70, 80, 90, 100]
    # Ws = [90, 100]
    counter = 0
    all_futures = []
    conf = {
            'use_solv': True,
            'desired_entropy': 1,
            'pop_size': 100,
            'num_gens': 100,
            'use_novelty': False
        }
    for w in Ws:
        counter += 1
        all_futures += one_run(counter, level_size=w, **conf)
    print(f"At end, we have {counter} runs to do. Running Experiment 104" )
    ray.get(all_futures)
    print("Done")

if __name__ == '__main__':
    experiment_107a();