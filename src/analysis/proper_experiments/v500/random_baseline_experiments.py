import os
import sys
import ray
import wandb
from baselines.random_baseline import RandomBaseline
from common.types import Verbosity
from common.utils import get_date
from experiments.config import Config
from experiments.experiment import Experiment
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
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

"""
    This runs a random baseline.
"""

def v_581_maze_random():
    ray.init()
    name = 'experiment_581_a'
    game = 'Maze'
    method = 'RandomBaseline'
    date = get_date()
    
    results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/'
    args = {}
    
    def get_pop():
        level = MazeLevel.random()
        game = MazeGame(level)
        return RandomBaseline(game, level)

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
        metrics_ =  [
            AStarSolvabilityMetric(proper_game, parent),
            CompressionDistanceMetric(proper_game),
            LeniencyMetric(proper_game),
            AStarDiversityMetric(proper_game, parent),
            AStarDifficultyMetric(proper_game, parent),
            AStarEditDistanceDiversityMetric(proper_game, parent),
        ]
        experiment = Experiment(config, get_pop, metrics_, log_to_wandb=True, log_to_dict=False, verbose=Verbosity.PROGRESS if seed == 0 else Verbosity.NONE)

        experiment.do_all()
        wandb.finish()

        return f"Completed with seed = {seed}"
    
    futures = [single_func.remote(i) for i in range(5)]
    print(ray.get(futures))

def v_582_mario_random():
    ray.init()
    name = f'582_a'
    game = 'Mario'
    method = 'RandomBaseline'
    message = f'Mario random baseline'
    date = get_date()
    
    results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/'
    mario_game = MarioGame(MarioLevel())
    
    args = {
        'message': message,
    }
    
    def get_pop():
        level = MarioLevel()
        game = MarioGame(level)
        return RandomBaseline(game, level)

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
        parent = AStarDiversityAndDifficultyMetric(proper_game, number_of_times_to_do_evaluation = 5)
        _metrics = [
            AStarSolvabilityMetric(proper_game, parent),
            CompressionDistanceMetric(proper_game),
            CompressionDistanceMetric(proper_game, use_combined_features=True),
            CompressionDistanceMetric(proper_game, do_mario_flat=True),
            LeniencyMetric(proper_game),
            AStarDiversityMetric(proper_game, parent),
            AStarDifficultyMetric(proper_game, parent),
            AStarEditDistanceDiversityMetric(proper_game, parent),
        ]

        experiment = Experiment(config, get_pop, _metrics, log_to_wandb=False, log_to_dict=True, verbose=Verbosity.PROGRESS if seed == 0 else Verbosity.NONE)

        experiment.do_all()
        wandb.finish()

        return f"Completed with seed = {seed}"
    
    futures = [single_func.remote(i) for i in range(5)]
    ray.get(futures)


if __name__ == '__main__':
    DO_MAZE = int(sys.argv[-1])
    if DO_MAZE:
        v_581_maze_random()
    else:
        v_582_mario_random()
