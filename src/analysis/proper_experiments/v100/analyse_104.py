from collections import defaultdict
import glob
import os
import pickle
import re

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from common.utils import get_latest_folder
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel

from metrics.rl.tabular.rl_agent_metric import RLAgentMetric
from metrics.rl.tabular.rl_difficulty_metric import RLDifficultyMetric
from novelty_neat.maze.neat_maze_level_generation import GenerateMazeLevelsUsingTiling
import seaborn as sns

def pretty_key(k):
    if 'time' in k:
        return ' '.join(map(str.title, k.split("_"))) + " (s)"
    # Thanks :) https://stackoverflow.com/a/37697078
    splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', k)).split()
    return ' '.join(splitted)

def analyse_104_with_line_graph():
    """
        This plots a line graph of experiment 104, as well as using some data from experiment 107.
        The x-axis will be level size and the y-axis the metrics, specifically time and maybe some others.
    """
    with open(get_latest_folder('../results/experiments/104b/runs/*/data.p'), 'rb') as f:
        data = pickle.load(f)

    
    def get_mean_standard_for_one_point_in_directga(width, mean_dic, std_dic):
        path = os.path.join(get_latest_folder(f'../results/experiments/experiment_107_a/Maze/DirectGA/2*'), f'{width}/*/*/*/*/*/*/*.p')

        li = glob.glob(path)
        print(len(li), path)
        assert len(li) == 5
        metrics = defaultdict(lambda: [])
        all_levels = []
        for p in li:
            with open(p, 'rb') as f:
                d = pickle.load(f)
                for key in d['eval_results_single']:
                        metrics[key].append(d['eval_results_single'][key])
                for key in ['generation_time']:
                    # DON'T divide by 100 here, as this was for 1 level. The experiment.py already normalised it.
                    metrics[key].append(d[key])
                all_levels.append(d['levels'][0])
        
        dir = f'results/maze/104/line_graph/levels_direct_ga'
        for i, l in enumerate(all_levels):
            os.makedirs(dir, exist_ok=True)
            plt.figure(figsize=(20, 20))
            plt.imshow(1 - l.map, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(os.path.join(dir, f'{width}-{i}.png'), pad_inches=0.1, bbox_inches='tight')
            plt.close()
        print("Direct ", metrics.keys())
        for key in metrics:
            metrics[key] = np.array(metrics[key])
            mean_dic[key].append(np.mean(metrics[key]))
            std_dic[key].append(np.std(metrics[key]))            

    D = data['data']
    # D[14] = data['original']
    fs = data['files']
    og_metrics = defaultdict(lambda: 0)
    for T in data['original']:
        things = T['eval_results_single']
        for key in things:
            og_metrics[key] += np.mean(things[key])

    for key in og_metrics:
        og_metrics[key] /= len(fs)
    all_metrics = {
        # 14: og_metrics
    }
    all_values_mean = defaultdict(lambda : [])
    all_values_std = defaultdict(lambda : [])
    
    all_values_mean_direct_ga = defaultdict(lambda : [])
    all_values_std_direct_ga = defaultdict(lambda : [])

    directga_widths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for w in directga_widths:
        get_mean_standard_for_one_point_in_directga(w, all_values_mean_direct_ga, all_values_std_direct_ga)
    widths = []
    the_keys_to_use = sorted(D.keys())
    for width in the_keys_to_use:
        levels_to_plot = []
        metrics = defaultdict(lambda: [])
        widths.append(width)
        for d in D[width]:
            levels_to_plot.append(d['levels'][0])

            for key in d['eval_results_single']:
                metrics[key].append(d['eval_results_single'][key])
            for key in ['generation_time']:
                if width != 14:
                    # for 14, it was measured properly.
                    # the values in here were for all levels, so we norm it to one level.
                    metrics[key].append(d[key] / 100)
                else:
                    metrics[key].append(d[key])
        
        for key in metrics:
            metrics[key] = np.array(metrics[key])
            all_values_mean[key].append(np.mean(metrics[key]))
            all_values_std[key].append(np.std(metrics[key]))

        dir = 'results/maze/104/line_graph/levels'
        os.makedirs(dir, exist_ok=True)
        for i, l in enumerate(levels_to_plot):
            l: MazeLevel
            plt.figure(figsize=(20, 20))
            l.show(True)
            plt.axis('off')
            plt.savefig(os.path.join(dir, f'{width}-{i}.png'), pad_inches=0.1, bbox_inches='tight')
            plt.close()
            
    metrics_to_plot = [
        'generation_time',
        'SolvabilityMetric',
        'CompressionDistanceMetric',
        'AStarDiversityMetric',
        'AStarDifficultyMetric',
        'AStarEditDistanceDiversityMetric'
    ]
    print("KEYS: ", all_values_mean.keys())
    sns.set_theme()
    for key in metrics_to_plot:
        
        all_values_mean[key] = np.array(all_values_mean[key])
        all_values_std[key] = np.array(all_values_std[key])

        all_values_mean_direct_ga[key] = np.array(all_values_mean_direct_ga[key])
        all_values_std_direct_ga[key] = np.array(all_values_std_direct_ga[key])

        plt.figure()
        plt.plot(widths, all_values_mean[key], label='NoveltyNEAT (Ours)')
        plt.fill_between(widths, all_values_mean[key] - all_values_std[key], all_values_mean[key] + all_values_std[key], alpha=0.5)
        if len(all_values_mean_direct_ga[key]) == 0:
            print(f"KEY = {key} does not have data for DirectGA")
        else:
            plt.plot(directga_widths, all_values_mean_direct_ga[key], label='DirectGA+')
            plt.fill_between(directga_widths, all_values_mean_direct_ga[key] - all_values_std_direct_ga[key], all_values_mean_direct_ga[key] + all_values_std_direct_ga[key], alpha=0.5)

        plt.xlabel("Level Width = Height")
        pkey = pretty_key(key).replace("Metric", '')
        plt.ylabel(pkey)
        plt.title(f"Comparing {pkey} vs Level Size. Higher is better.")
        if 'time' in key.lower():
            plt.title(f"Comparing {pkey} vs Level Size. Lower is better.")
            plt.scatter([14, 20], [40000, 70000], marker='x', color='red', label='PCGRL (Turtle)')
            plt.yscale('log')
        plt.tight_layout()
        # plt.show()
        plt.legend()

        plt.savefig(f'results/maze/104/line_graph/{key}.png')


    df = pd.DataFrame(all_metrics).round(2)
    print(df)


if __name__ == '__main__':
    analyse_104_with_line_graph()