"""
    This analyses the results from v106_metrics.py
"""

from collections import defaultdict
from email.policy import default
import os
import pickle
from typing import Dict, List
from matplotlib import pyplot as plt

import numpy as np
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.rl.tabular.rl_agent_metric import RLAgentMetric
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric
from metrics.diversity.simple_diversity import EditDistanceMetric, HammingDistanceMetric
from common.utils import get_only_solvable_metrics
from scipy.stats import pearsonr


def main(compare_now = [
        'LeniencyMetric',
        # 'RLDifficultyMetric',
        'RLDifficultyMetric',
        'AStarDifficultyMetric'
    ], name='difficulty', file_to_use=None):
    """
        Analyse experiment 106, the one that ran our metrics and others, so we can compare.
    """
    path = file_to_use

    points = defaultdict(lambda: [])

    def add_in_other_metrics(l, levels):
        g = MazeGame(MazeLevel())
        editdist = EditDistanceMetric(g)
        l['EditDistanceMetric'] = editdist.evaluate(levels)
        
        hamming = HammingDistanceMetric(g)
        l['HammingDistanceMetric'] = hamming.evaluate(levels)




    with open(path, 'rb') as f:
        ls = pickle.load(f)
        for l in ls:
            with open(l['name'], 'rb') as level_file:
                levels = pickle.load(level_file)['levels']
            add_in_other_metrics(l, levels)


            for key in compare_now:
                if name == 'difficulty':
                    is_solvable = np.array(l['SolvabilityMetric']) > 0
                    vals = np.array(l[key])[is_solvable]
                else:

                    if type(l[key]) == list or getattr(l[key], '__iter__', False) or type(l[key]) == np.ndarray:
                        vals = get_only_solvable_metrics(l[key], np.array(l['SolvabilityMetric'])) 
                    else:
                        vals = np.array(l[key])
                points[key].append((np.mean(vals), np.std(vals)))
    for i in range(len(compare_now)):
        x = points[compare_now[i]]
        x_mean, x_std = zip(*x)
        x_mean = np.array(x_mean)
        x_std = np.array(x_std)
        I = np.arange(len(x_mean))
        plt.plot(x_mean, label=compare_now[i])
        plt.fill_between(I, x_mean - x_std, x_mean + x_std, alpha=0.5)

    plt.xlabel("Run index")
    plt.ylabel("Metric Values")
    plt.title(f"Comparing {name} metrics")
    plt.legend()

    dir = './results/maze/106'
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, name + ".png"))
    plt.close()
    if len(compare_now) == 2:
        x = points[compare_now[0]]
        x_mean, x_std = zip(*x)
        I = np.argsort(x_mean)
        
        y = points[compare_now[1]]
        y_mean, y_std = zip(*y)
        r, p = pearsonr(x_mean, y_mean)
        plt.plot(np.array(x_mean)[I], np.array(y_mean)[I])
        plt.title(f"R={r}, p={p}")
        plt.savefig(os.path.join(dir, name + "_correlation" + ".png"))

if __name__ == '__main__':
    F_ = main
    F_(
        [
        'LeniencyMetric',
        'AStarDifficultyMetric'
    ]
    );
    plt.close()
    F_([
        'CompressionDistanceMetric',
        'EditDistanceMetric', 'HammingDistanceMetric',
        'AStarEditDistanceDiversityMetric',
        'AStarDiversityMetric',
    ], 'diversity')
    plt.close()
    F_([
        'CompressionDistanceMetric',
        'AStarEditDistanceDiversityMetric',
        'AStarDiversityMetric',
    ], 'diversity-lite')