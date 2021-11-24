"""
    This strives to compare metrics to each other, specifically
    CD vs A* div
        -> As well as CD on its own.
    Len vs A* diff.


    Uses data and techniques from analyse_106 & analyse_202
"""


from collections import defaultdict
import glob
import os
import pickle
from typing import List
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from analysis.proper_experiments.v400.analyse_all_statistical_tests import clean_metric
from common.utils import get_latest_folder, get_only_solvable_metrics
from games.level import Level
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
from games.maze.maze_game import MazeGame

from games.maze.maze_level import MazeLevel
from metrics.a_star.a_star_metrics import AStarDiversityAndDifficultyMetric
from metrics.diversity.simple_diversity import EditDistanceMetric, HammingDistanceMetric
from scipy.signal import savgol_filter
import seaborn as sns

from metrics.horn.compression_distance import CompressionDistanceMetric
sns.set_theme()
from scipy.stats import pearsonr
from metrics.rl.tabular.rl_agent_metric import compare_actions_edit_distance

MAZE_FILE_TO_USE = get_latest_folder('../results/experiments/106/runs/*/data.p')
MARIO_202_METRIC_DATA  = get_latest_folder('../results/experiments/experiment_202/runs/*/data.p')


MARIO_206_DATA = get_latest_folder('../results/experiments/206a/runs/*/data.p')
MARIO_206B_DATA = get_latest_folder('../results/experiments/experiment_206b/Mario/DirectGA/2*')

# not latest, for metrics
V104_DATA = '../results/experiments/104b/runs/2021-10-29_15-37-53/data.p'

def get_all_values_maze(return_test_parents=False):
    """
    Uses 106
    """
    FILE = file_to_use = MAZE_FILE_TO_USE
    path = file_to_use

    points = defaultdict(lambda: [])
    parents = defaultdict(lambda: [])

    def add_in_other_metrics(l, levels):
        g = MazeGame(MazeLevel())
        # editdist = EditDistanceMetric(g)
        # l['EditDistanceMetric'] = editdist.evaluate(levels)
        
        hamming = HammingDistanceMetric(g)
        l['HammingDistanceMetric'] = hamming.evaluate(levels)

    bads = ['name', 'AveragePairWiseDistanceMetric', 'test_parent']
    with open(path, 'rb') as f:
        ls = pickle.load(f)
        for l in ls:
            with open(l['name'], 'rb') as level_file:
                levels = pickle.load(level_file)['levels']
            add_in_other_metrics(l, levels)
            for key in l:
                if key == 'test_parent' or key == 'name':
                    parents[key].append(l[key])
                if key in bads: continue
                if type(l[key]) == list or getattr(l[key], '__iter__', False) or type(l[key]) == np.ndarray:
                    vals = get_only_solvable_metrics(l[key], np.array(l['SolvabilityMetric'])) 
                else:
                    print(f"KEY = {key} is bad", l[key])
                    assert False
                points[key].append({'mean': np.mean(vals), 'std': np.std(vals), 'all':vals})
    if return_test_parents:
        return points, parents
    return points

def get_mario_202_data(return_alls2=False, return_parents=False):
    alls = defaultdict(lambda: [])
    alls_2 = defaultdict(lambda: [])
    parents = defaultdict(lambda: [])
    with open(MARIO_202_METRIC_DATA, 'rb') as f:
        li = pickle.load(f)
        for index, dic in enumerate(li):
            for key in dic:

                if key in ['AveragePairWiseDistanceMetric', 'test_parent', 'name']: 
                    parents[key].append(dic[key])
                    continue
                if key != "SolvabilityMetric":
                    vals = get_only_solvable_metrics(dic[key], dic['SolvabilityMetric'])
                else:
                    vals = dic[key]
                alls[key].extend(vals)
                alls_2[key].append(np.mean(vals))
    if return_alls2:
        return alls, alls_2
    if return_parents:
        return alls, parents
    return alls

mapping = {
    'CompressionDistanceMetric': "Concatenated",
    'CompressionDistanceMetric_FlatMario': "Flat",
    'CompressionDistanceMetric_CombinedFeatures': "Normal"
}

def v452_b_cd_size_dependent_with_direct_ga_get_data():
    dic_to_return = {}
    P = MARIO_206B_DATA
    for width in [28, 56, 85, 114, 171, 228]:
        path = f'{MARIO_206B_DATA}/{width}/*/*/*/*.p'
        li = glob.glob(path)
        assert len(li) == 5
        metrics = defaultdict(lambda: [])
        all_levels = []
        for p in li:
            little_d = {}
            with open(p, 'rb') as f:
                d = pickle.load(f)
                for key in d['eval_results_single']:
                    V = d['eval_results_all'][key]
                    little_d[key] = d
                for key in ['generation_time']:
                    little_d[key] = d[key]
            all_levels.append(d)
        dic_to_return[width] = all_levels
    return dic_to_return

def v452_check_cd_as_function_of_size():
    import seaborn as sns
    # uses data from experiment 104 & 107, to analyse CD's
    # degeneration as the level size increases.
    for game in ['maze', 'mario', 'mario_directga']:
        DIR = f'results/v400/metrics/{game}'
        P = V104_DATA if game == 'maze' else MARIO_206_DATA
        if game == 'mario_directga':
            dic = {'data': v452_b_cd_size_dependent_with_direct_ga_get_data()}
        else:
            with open(P, 'rb') as f:
                dic = pickle.load(f)
        
        print(dic.keys())
        data = dic['data']
        ls = list(data.keys())[0]
        for M in set(data[ls][0]['eval_results_all'].keys()) - {'AveragePairWiseDistanceMetric'}:
            alls = {}
            cols = ['red', 'green', 'blue', 'orange']
            unique = list(data.keys())
            palette = dict(zip(unique, sns.color_palette(n_colors=len(unique))))
            print(f"For M = {M}")
            big_dic = {M: [], 'Width': []}
            for i, (w, list_of_seeds) in enumerate(data.items()):
                all_d = []
                for seed_data in list_of_seeds:
                    solvs = seed_data['eval_results_all']['SolvabilityMetric']
                    corrected = get_only_solvable_metrics(seed_data['eval_results_all'][M], solvs)
                    all_d += list(corrected)
                # , kde=True, stat='percent'
                print(f"\tW = {w}, all_d length = {len(all_d)}")
                big_dic['Width'].extend([w] * len(all_d))
                big_dic[M].extend(all_d)
                alls[w] = all_d
            M_TO_SAVE = M
            print("Doing", M, game)
            
            sns.histplot(big_dic, x=M, hue=f'Width', palette=palette)
            if "CompressionDistanceMetric" in M and 'mario' in game:
                M = f"CD: {mapping[M]}"
            plt.xlabel(clean_metric(M))
            plt.title(f"Distribution of {clean_metric(M)} for various level sizes - {game.title()}")
            dir = os.path.join(DIR, 'cd_bad')
            os.makedirs(dir, exist_ok=True)
            plt.savefig(os.path.join(dir, f'{M_TO_SAVE}_distribution.png'))
            plt.close()

def v453_cd_dependent_on_representation():
    alls = get_mario_202_data()
    DIR = f'results/v400/metrics/mario/202_cd_repr/'
    for M, val in alls.items():
        sns.histplot(val)
        plt.xlabel(clean_metric(M))
        plt.title(f"Distribution of {clean_metric(M)} - Mario")
        plt.legend()
        dir = DIR
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, f'{M}_distribution.png'))
        plt.close()
    
    keys_to_consider = [
        'CompressionDistanceMetric_CombinedFeatures',
        'CompressionDistanceMetric',
        'CompressionDistanceMetric_FlatMario',
    ]

    palette = dict(zip(map(lambda M: clean_metric(mapping[M]), keys_to_consider), sns.color_palette(n_colors=len(keys_to_consider))))

    all_vals = {
        'Representation': [],
        'Metric Values': []
    }
    for M in keys_to_consider:
        CM = clean_metric(mapping[M])
        all_vals['Representation'].extend([CM] * len(alls[M]))
        all_vals['Metric Values'].extend(alls[M])
    
    sns.histplot(all_vals, x='Metric Values', hue='Representation', label=CM, palette=palette)
    plt.xlabel("Metric Values")
    plt.title(f"Distribution of Compression Distance metrics - Mario")
    # plt.legend()
    dir = DIR
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, f'compression_distance_comparing_distribution.png'))
    plt.close()
    dic = {}
    for M1 in keys_to_consider:
        CM1 = clean_metric(mapping[M1])
        dic[CM1] = {}
        for M2 in keys_to_consider:
            CM2 = clean_metric(mapping[M2])
            R, p = pearsonr(alls[M1], alls[M2])
            dic[CM1][CM2] = R
            print(f"FOR {M1:<50} & {M2:<50}, R={R:<20} | p={p}")
    df = (pd.DataFrame(dic))
    print(df)
    df.to_latex(os.path.join(DIR, "correlations.tex"), escape=False)

def v461_len_vs_astar_correlation():
    # Correlation twixt leniency and astar diff.
    mode = 'mario'
    for mode in ['maze', 'mario']:
        # mode = 'maze'
        
        points = get_mario_202_data(True) if mode == 'mario' else get_all_values_maze()
        compare_now = ['LeniencyMetric', 'AStarDifficultyMetric']
        name = 'difficulty'
        As = []
        if mode =='mario':
            points, _ = points
        for i in compare_now:
            xs = (points[i])
            if mode == 'maze':
                x_mean, x_std, x_all = zip(*[(x['mean'], x['std'], x['all']) for x in xs])
                x_mean = np.array(x_mean)
                x_std = np.array(x_std)
                x_mean = np.array(sum(map(list,x_all), []))
            else:
                x_mean = np.array(xs)
                print(x_mean)
            As.append(x_mean)
            I = np.arange(len(x_mean))
            plt.plot(x_mean, label=i)
        
        DIR = 'results/v400/metrics/correlation_leniency'
        os.makedirs(DIR, exist_ok=True)
        plt.xlabel("Run index")
        plt.ylabel("Metric Values")
        plt.title(f"Comparing {name} metrics")
        plt.legend()
        # plt.show()    
        plt.close()
        Is = np.argsort(As[0])
        plt.plot(As[0][Is], As[1][Is])
        r, p = pearsonr(As[0][Is], As[1][Is])
        plt.xlabel(compare_now[0])
        plt.ylabel(compare_now[1])
        plt.title(f"R = {r}, P = {p} over {len(As[0])} levels")
        plt.savefig(os.path.join(DIR, f"corr_{mode}.png"))
        plt.close()


def _compare_correlation_between_astar_and_edit_dist_astar(name='correlation_astar_metrics_edit', all_names = ['AStarDiversityMetric', 'AStarEditDistanceDiversityMetric', 'CompressionDistanceMetric']):
    # This wants to compare a* edit distance vs a* sampled dist,
    # for both mario and maze.
    mode = 'maze'
    should_use_alls = True
    DIR = f'results/v400/metrics/{name}'
    os.makedirs(DIR, exist_ok=True)

    things = defaultdict(lambda: plt.subplots(2, 2, figsize=(20, 20)))
    for mode in ['maze', 'mario']:
        for should_use_alls in [True, False]:
            if mode == 'maze':
                alls = get_all_values_maze()

                alls = {
                    k: (
                        sum([[(li['mean'])] for li in v], [])
                        if should_use_alls 
                        else 
                        sum([list(li['all']) for li in v], []))  
                        for k, v in alls.items()
                }
                # alls = sum([i['alls'] for i in ], [])
            else:
                alls, alls2 = get_mario_202_data(True)
                if should_use_alls:
                    alls = alls2
            if mode == 'mario' and name == 'correlation_astar_metrics_edit':
                all_names.append("CompressionDistanceMetric_FlatMario")
            index = (mode == 'mario') * 2 + (should_use_alls)
            _all_names = sorted(set(all_names + ['CompressionDistanceMetric_FlatMario']))
            palette = dict(zip(_all_names, sns.color_palette(n_colors=4)))
            for i in range(len(all_names)):
                for j in range(i+1, len(all_names)):
                    names = [all_names[i], all_names[j]]
                    name_to_save_as = '__'.join(names)
                    ax = things[name_to_save_as][1].ravel()[index]
                    S = np.argsort(alls[names[0]])
                    for n in names:
                        # plt.plot(np.array(alls[n])[S], label=n)
                        ax.plot(np.array(alls[n])[S], label=n, color=palette[n])
                        # sns.lineplot(data={'data': np.array(alls[n])[S], 'abc': n}, x='data', label=n, palette=palette, hue='abc', ax=ax)
                    r, p = pearsonr(alls[names[0]], alls[names[1]])
                    ax.set_title(f'Pearson\'s R = {np.round(r, 3)} | P = {p:1.2e}. {mode} | {"Alls" if not should_use_alls else "Means"} | Over {len(S)} points')
                    ax.set_xlabel('Level Index / Group Index')
                    ax.set_ylabel('Metric')
                    ax.legend()
    for key, (fig, axs) in things.items():
        fig.savefig(os.path.join(DIR, f'{key}.png'))
        plt.close()


if __name__ == '__main__':
    v453_cd_dependent_on_representation()
    v452_check_cd_as_function_of_size()

    v461_len_vs_astar_correlation()
    _compare_correlation_between_astar_and_edit_dist_astar()
    _compare_correlation_between_astar_and_edit_dist_astar('leniency_correlations', ['LeniencyMetric', 'AStarDifficultyMetric'])
