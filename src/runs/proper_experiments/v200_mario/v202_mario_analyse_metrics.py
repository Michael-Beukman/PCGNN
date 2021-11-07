import glob
import os
import pickle
import pprint
from typing import List
import ray
from common.utils import get_date
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
from games.maze.maze_game import MazeGame

from games.maze.maze_level import MazeLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarEditDistanceDiversityMetric, AStarSolvabilityMetric
from metrics.average_pairwise_distance import AveragePairWiseDistanceMetric
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.metric import Metric
from metrics.rl.tabular.rl_agent_metric import RLAgentMetric
from metrics.rl.tabular.rl_difficulty_metric import RLDifficultyMetric
from metrics.solvability import SolvabilityMetric
from analysis.proper_experiments.v200.analyse_202 import main as analyse_these_results

def main():
    """
        Tries to run metrics on many levels to get a feel for them, as well as plotting all the things
        Step 1: Some of my things

        Step 2: Some of the original levels.
    """
    
    ray.init()

    # What levels ?
    
    pickles = \
        glob.glob('../results/v14_mario_neat_island/Mario/NeatNoveltyIsland/2021-09-23_07-37-38/50/5/*/*.p') + \
        glob.glob('../results/v10_mario_neat/Mario/NeatNovelty/2021-09-07_06-01-28/50/500/*/*.p') + \
        glob.glob('../results/experiments/experiment_204e/Mario/NeatNovelty/2021-10-09_06-36-48/100/150/True/1/0/6/2/-1/*/*.p') +\
        glob.glob('../results/experiments/experiment_201_b/Mario/DirectGA/2021-09-26_09-57-35/100/100/0.0/0.0/1.0/0.5/114/20/10/10/2/2/*/*.p')+ \
        glob.glob('../results/experiments/experiment_201_a/Mario/DirectGA/2021-09-21_07-42-10/10/50/0.5/0.5/0.5/0.5/20/20/10/40/2/2/*/*.p') + \
        glob.glob('../results/experiments/experiment_201_c/Mario/DirectGA/2021-10-12_06-54-30/10/50/0.5/0.5/0.5/0.5/20/20/10/40/2/2/use_novelty/*/*.p')

    pprint.pprint(pickles)

    print("LEN = ", len(pickles))
    game = MarioGame(MarioLevel())

    def get_metrics(g): 
        parent = AStarDiversityAndDifficultyMetric(g, number_of_times_to_do_evaluation=5)

        div = AStarDiversityMetric(g, parent)
        diff = AStarDifficultyMetric(g, parent)
        solv = AStarSolvabilityMetric(g, parent)
        return [
        # SolvabilityMetric(g),
        solv,
        LeniencyMetric(g),
        CompressionDistanceMetric(game),
        CompressionDistanceMetric(game, use_combined_features=True),
        CompressionDistanceMetric(game, do_mario_flat=True),
        AveragePairWiseDistanceMetric(g),
        div,
        diff,
        AStarEditDistanceDiversityMetric(g, parent)
    ], parent


    @ray.remote
    def single_func(index: int, pickle_name: str):
        metrics, _parent = get_metrics(game)
        metrics: List[Metric]
        mydic = {
            'name': pickle_name
        }
        with open(pickle_name, 'rb') as f:
            ans = pickle.load(f)
            levels = ans['levels']
        for k, metric in enumerate(metrics):
            print(f"Index {index} starting with metric {metric.name()} {k+1} / {len(metrics)}")
            mydic[metric.name()] = metric.evaluate(levels)
        mydic['test_parent'] = _parent
        return mydic

    futures = [single_func.remote(i, pickles[i]) for i in range(len(pickles))]
    all_values = (ray.get(futures))

    d = f'../results/experiments/experiment_202/runs/{get_date()}'
    os.makedirs(d, exist_ok=True)
    FILENAME = f"{d}/data.p"
    with open(FILENAME, 'wb+') as f:
        pickle.dump(all_values, f)
    return FILENAME


if __name__ == '__main__':
    FILENAME = main()

    import matplotlib.pyplot as plt
    analyse_these_results(file_to_use=FILENAME)
    plt.close()
    analyse_these_results([
        'CompressionDistanceMetric',
        'CompressionDistanceMetric_FlatMario',
        'CompressionDistanceMetric_CombinedFeatures',
        # 'EditDistanceMetric', 
        'HammingDistanceMetric',
        'AStarDiversityMetric',

    ], 'diversity', file_to_use=FILENAME)