from collections import defaultdict
import os
import pickle
from matplotlib import pyplot as plt

import numpy as np
from common.utils import get_only_solvable_metrics
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.rl.tabular.rl_agent_metric import RLAgentMetric
from metrics.diversity.simple_diversity import EditDistanceMetric, HammingDistanceMetric
from scipy.stats import pearsonr


def main(compare_now = [
        'LeniencyMetric',
        'AStarDifficultyMetric'
    ], name='difficulty', file_to_use=None):
    """
        Analyse v202, the metrics on Mario

    """
    path = file_to_use # FILE

    points = defaultdict(lambda: [])

    def add_in_other_metrics(l, levels):
        print("Here")
        # return
        g = MarioGame(MarioLevel())
        editdist = EditDistanceMetric(g)
        # l['EditDistanceMetric'] = editdist.evaluate(levels)
        
        hamming = HammingDistanceMetric(g)
        l['HammingDistanceMetric'] = hamming.evaluate(levels)


    with open(path, 'rb') as f:
        ls = pickle.load(f)
        for l in ls:
            with open(l['name'], 'rb') as level_file:
                levels = pickle.load(level_file)['levels']
            add_in_other_metrics(l, levels)


            for key in compare_now:
                if type(l[key]) == list:
                    vals = get_only_solvable_metrics(l[key], np.array(l['SolvabilityMetric'])) 
                    if key == 'AStarDifficultyMetric':
                        vals = np.array(vals)
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
    K = ', '.join(compare_now[:2]) + '\n' + ', '.join(compare_now[2:])
    K = f"{name} metrics on Mario"
    plt.title(f"Comparing {K}")
    plt.legend()
    plt.tight_layout()

    dir = './results/mario/202'
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
