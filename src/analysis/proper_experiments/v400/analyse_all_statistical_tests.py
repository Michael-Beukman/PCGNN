"""
    This file attempts to be a central place where we compare different results (from both Mario and Maze)
    for different metrics, using different statistical tests, etc.
"""
from collections import defaultdict
from email.policy import default
import glob
import os
import pickle
import pprint
import re
import sys
from typing import Any, Dict, List
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from common.utils import get_only_solvable_metrics, get_latest_folder
from experiments.config import Config
from games.game import Game
from games.level import Level
from games.mario.mario_level import MarioLevel
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric
import matplotlib.patches as mpatches
from scipy.stats import shapiro, mannwhitneyu, ttest_ind

from novelty_neat.maze.a_star import do_astar_from_level

DIR_TO_SAVE = 'results/v400/methods/'
os.makedirs(DIR_TO_SAVE, exist_ok=True)

DEFAULT_MONIKER = 'NoveltyNEAT (Ours)'


MARIO_RESULTS_FILE = get_latest_folder('results/mario/metrics_and_levels/*/data.p')
MAZE_RESULTS_FILE = get_latest_folder('results/maze/metrics_and_levels/*/data.p')

MARIO_202_RUNS = get_latest_folder('../results/experiments/experiment_202/runs/*/data.p')
MAZE_106_METRICS_FILE = get_latest_folder('../results/experiments/106/runs/*/data.p')


def pretty_key(k):
    if 'time' in k:
        return ' '.join(map(str.title, k.split("_"))) + " (s)"
    return k


def clean_metric(m):
    if 'AStarEditDistanceDiversityMetric' in m: return m.replace('AStarEditDistanceDiversityMetric', "A* Diversity (Edit)")
    m = m.replace('Metric', '').replace("AStar", "A*")
    splitted = re.sub('([A-Z][a-z]+)', r' \1',
                      re.sub('([A-Z]+)', r' \1', m)).split()
                      
    ans = ' '.join(splitted)

    if 'time' in ans:
        return ' '.join(map(str.title, ans.split("_"))) + " (s)"
    return ans


def mean_std(v, decimals = 2):
    if 'all' not in v:
        v['all'] = []
    return better(v, decimals=decimals)[0]
    m, s = v['mean'], v['std']
    m = np.round(m, 2)
    s = np.round(s, 2)
    return f"{m} ({s})"


def make_nice(s):
    name = s
    if 'neat_best_0815' in s or 'experiment_105' in s or 'experiment_204e' in s:
        return 'NoveltyNEAT (Ours)'

    if '201_a' in name:
        return "DirectGA+"
    if '201_b' in name:
        return "DirectGA"  # Default
    if '201_d' in name:
        return "DirectGA (Novelty)"
    if 'pcgrl' in name and 'wide' in name:
        return "PCGRL (Wide)"
    if 'pcgrl' in name and 'turtle' in name:
        return "PCGRL (Turtle)"
    if '102_d' in name:
        return None  # 'DirectGA With NoveltyV2'
    if '102_f' in name:
        return 'DirectGA (Novelty)'

    if '102_aaa' in name and 'True/True' in name:
        return None  # 'Optimised DirectGAV2'
    
    if '102_aaa' in name and 'True/False' in name and '2021-09-16_07-51-09' not in name:
        return 'DirectGA+'  # Optimised
    elif '102_aaa' in name and 'True/False' in name and '2021-09-16_07-51-09' in name:
        # Old one
        return None

    K = ' '.join(map(str.title, s.split("_")))
    K = K.replace("Directga", "DirectGA")
    K = K.replace("Withnovelty", "With Novelty")
    K = K.replace("Perfectsolve", "")
    K = K.strip()
    if K == "DirectGA":
        K = "Optimised DirectGAV3"
    return K


def order_columns_of_df(df, rows=False):
    ordering = [
        (0, DEFAULT_MONIKER),
        (1, "DirectGA"),
        (2, "DirectGA+"),
        # (2, "Optimised DirectGAV2"),
        (3, "DirectGA (Novelty)"),
        # (3, "DirectGA With NoveltyV2"),
        (4, "PCGRL (Wide)"),
        (5, "PCGRL (Turtle)"),
    ]
    if rows:
        print(df.index)
        return df.reindex([o[1] for o in ordering])
    else:
        cols = list(df.columns)

        ordered = [i[1] for i in ordering if i[1] in cols]
        assert len(ordered) == len(cols)

        # i = cols.index(DEFAULT_MONIKER)
        # cols = [DEFAULT_MONIKER] + cols[:i] + cols[i+1:]
        df = df[ordered]
        return df


def clean_the_single_results_of_one_run(one_run):
    n = 'eval_' if 'eval_results_all' in one_run else ''
    if n == '':
        # Mario
        for K in range(len(one_run['results_all']['SolvabilityMetric'])):
            solvs = one_run['results_all']['SolvabilityMetric'][K]
            for key, _ in one_run['results_all'].items():
                if key == 'SolvabilityMetric':
                    continue
                li = one_run['results_all'][key][K]
                if len(one_run['results_all'][key]) != 5:
                    print(key, len(one_run['results_all']
                          [key]), '\n' + "===" * 50)
                # assert len(one_run['results_all'][key]) == 5, f"len({key}) = {len(one_run['results_all'][key])} != 5"
                if 'time' in key or len(li) == 1:
                    continue
                new_li = get_only_solvable_metrics(li, solvs)
                new = np.mean(new_li)
                old = one_run['results_single'][key][K]
                one_run['results_single'][key][K] = new
                
                
                one_run['results_all'][key][K] = new_li
        return one_run
    else:
        # Cleans results and only shows solvable levels.
        solvs = one_run[f'{n}results_all']['SolvabilityMetric']
        for key, li in one_run[f'{n}results_all'].items():
            if key == 'SolvabilityMetric':
                continue
            if len(li) == 1:
                continue
            new = np.mean(get_only_solvable_metrics(li, solvs))
            old = one_run[f'{n}results_single'][key]
            one_run[f'{n}results_single'][key] = new
        return one_run


def better(v, decimals=2):
    m, s = v['mean'], v['std']
    if m > 2000 :
        def make_thing_better(s):
            return s
            a, b = s.split("e+")
            if b[0] == '0': b = b[1:]
            # return f"${a} \times 10^{b}$"
            return f"{a} \times 10^{b}"
        K = make_thing_better(f"{m:.0f}")#.replace(",", " ")
        P = make_thing_better(f"{s:.0f}")
        if np.round(s, 2) == 0:
            return f"{K} (0)", v['all']
        else:
            return f"{K} ({P})", v['all']
    old_m = m
    m = np.round(m, decimals)
    s = np.round(s, decimals)
    if s == 0:
        if old_m >= 0.002 and m == 0:
            s = f"{old_m:1.1e}"
            a, b = s.split("e-")
            if b[0] == '0': b = b[1:]
            b = '{-' + b + "}"
            return f"${a} \times 10^{b}$ (0)", v['all']
            return f"{m} (0)", v['all']

        return f"{m} (0)", v['all']
    else:
        if decimals == 2:
            return f"{m:.2f} ({s:.2f})", v['all']
        elif decimals == 3:
            return f"{m:.3f} ({s:.3f})", v['all']
        else:
            raise Exception("More decimals not supported")


def _add_to_overall_dics(overall_dic, overall_dic_for_stats, name, values):
    overall_dic[name] = {pretty_key(k): mean_std(v) for k, v in values.items()}
    overall_dic_for_stats[name] = {pretty_key(
        k): v['all'] for k, v in values.items()}


def get_all_results_from_methods_maze(return_levels=False, return_all_of_the_results_not_just_single=False):
    FILES = [
        MAZE_RESULTS_FILE
    ]

    overall_dic = {}

    def get_moniker(name):
        ans = make_nice(name)
        return ans

    def prett(metric_name):
        return ' '.join(metric_name.split("_"))
    all_dics = {}
    for FILE in FILES:
        with open(FILE, 'rb') as f:
            all_dics |= pickle.load(f)
    dic = all_dics
    overall_dic_for_stats_test = {}
    dic_of_levels = {}
    dic_of_all_results = {}
    if 1:
        # dic = pickle.load(f)
        for single_run in dic:
            moniker = get_moniker(single_run)
            if moniker == -1 or moniker is None:
                continue
            little_d = {}
            stats_d = {}
            dic[single_run] = clean_the_single_results_of_one_run(
                dic[single_run])
            single_results = dic[single_run]['results_single']
            
            results_all = dic[single_run]['results_all']

            for key in single_results:
                if len(single_results[key]) != 5:
                    print(f"LL of {moniker} with {key} is not 5")
                    exit()
            names = dic[single_run]['names']
            gentimes, traintimes = [], []
            levels = []
            for n in names:
                with open(n, 'rb') as f:
                    D = pickle.load(f)
                    gen_time = D.get('generation_time',
                                     D.get('time_per_level'))
                    train_time = D.get('train_time', -1)
                    gentimes.append(gen_time)
                    traintimes.append(train_time)
                    levels.append(D['levels'][0])
            dic_of_levels[moniker] = levels
            for key, li in single_results.items():
                mean, std = np.mean(li), np.std(li)
                little_d[prett(key)], stats_d[prett(key)] = better(
                    {'mean': mean, 'std': std, 'all': li})
            little_d['Generation Time (s)'], stats_d['Generation Time (s)'] = better(
                {'mean': np.mean(gentimes), 'std': np.std(gentimes), 'all': gentimes})
            little_d['Train Time (s)'], stats_d['Train Time (s)'] = better(
                {'mean': np.mean(traintimes), 'std': np.std(traintimes), 'all': traintimes})
            overall_dic[moniker] = little_d
            overall_dic_for_stats_test[moniker] = stats_d
            
            dic_of_all_results[moniker] = results_all
    if return_levels:
        return overall_dic, overall_dic_for_stats_test, dic_of_levels
    if return_all_of_the_results_not_just_single:
        return overall_dic, overall_dic_for_stats_test, dic_of_all_results
    return overall_dic, overall_dic_for_stats_test


def get_all_results_from_methods_mario(return_levels=False, return_all_of_the_results_not_just_single=False):
    FILES = [
        MARIO_RESULTS_FILE
    ]

    overall_dic = {}

    def get_moniker(name):
        return make_nice(name)

    def prett(metric_name):
        return ' '.join(metric_name.split("_"))
    all_dics = {}
    for FILE in FILES:
        with open(FILE, 'rb') as f:
            all_dics |= pickle.load(f)
    dic = all_dics
    overall_dic_for_stats_test = {}
    dic_of_levels = {}
    
    dic_of_all_results = {}

    if 1:
        # dic = pickle.load(f)
        for single_run in dic:
            moniker = get_moniker(single_run)
            little_d = {}
            stats_d = {}
            dic[single_run] = clean_the_single_results_of_one_run(
                dic[single_run])
            single_results = dic[single_run]['results_single']
            for key in single_results:
                if len(single_results[key]) != 5:
                    print(f"LL of {moniker} with {key} is not 5")
                    exit()
            dic_of_all_results[moniker] = dic[single_run]['results_all']

            dic_of_all_results[moniker]['train_time'] = []
            dic_of_all_results[moniker]['generation_time'] = []

            names = dic[single_run]['names']
            gentimes, traintimes = [], []
            levels = []
            for n in names:
                with open(n, 'rb') as f:
                    D = pickle.load(f)
                    gen_time = D.get('generation_time',
                                     D.get('time_per_level'))
                    train_time = D.get('train_time', -1)
                    gentimes.append(gen_time)
                    traintimes.append(train_time)
                    levels.append(D['levels'][0])
            dic_of_levels[moniker] = levels
            dic_of_all_results[moniker]['train_time'] = traintimes
            dic_of_all_results[moniker]['generation_time'] = gentimes
            for key, li in single_results.items():
                mean, std = np.mean(li), np.std(li)
                little_d[prett(key)], stats_d[prett(key)] = better(
                    {'mean': mean, 'std': std, 'all': li})
            little_d['Generation Time (s)'], stats_d['Generation Time (s)'] = better(
                {'mean': np.mean(gentimes), 'std': np.std(gentimes), 'all': gentimes})
            little_d['Train Time (s)'], stats_d['Train Time (s)'] = better(
                {'mean': np.mean(traintimes), 'std': np.std(traintimes), 'all': traintimes})

            overall_dic[moniker] = little_d
            overall_dic_for_stats_test[moniker] = stats_d
    if return_levels:
        return overall_dic, overall_dic_for_stats_test, dic_of_levels
    if return_all_of_the_results_not_just_single:
        return overall_dic, overall_dic_for_stats_test, dic_of_all_results
    return overall_dic, overall_dic_for_stats_test


def add_significance_to_dic_item(item: str, p: float, d: float, add_emblems: bool = True, is_highest=False):
    s = ""
    v = item
    # if is_highest: v = r"\textbf{" + item + "}"
    if add_emblems and p < 0.05:
        # s += "*"
        if p < 0.01:
            s += "*"
            if p < 0.001:
                s += "*"
        if abs(d) >= 0.8:
            s += "\dagger"
        
        v = r"\textbf{" + item + "}"
        
        v = "$" + v + "^{" + s + "}" + "$"
    return v


def general_thing(name_to_save_as: str, metric_names_to_use: List[str], alternative='two-sided'):
    test_str = ''
    is_all_normal = True
    if type(alternative) == str:
        alternatives = [alternative] * len(metric_names_to_use)
    else:
        alternatives = alternative
    dic_for_df = defaultdict(lambda: {})
    for game in ['Maze', "Mario"]:
        if game == 'Maze':
            dic_of_mean_stds, dic_of_all_values = get_all_results_from_methods_maze()
        else:
            dic_of_mean_stds, dic_of_all_values = get_all_results_from_methods_mario()
        for metric_name_to_use, alternative in zip(metric_names_to_use, alternatives):
            metric_values_to_compare_against = dic_of_all_values[DEFAULT_MONIKER].get(
                metric_name_to_use, [-1] * 5)
            normals = []
            data = {}
            how_many_true = 0
            for method_name, dic_of_metrics in dic_of_all_values.items():
                normals.append((shapiro(dic_of_metrics.get(
                    metric_name_to_use, [-1] * 5)).pvalue >= 0.05, method_name))
                if normals[-1][0]:
                    how_many_true += 1

            # is_all_normal = (min(normals)[0] == True) and is_all_normal
            is_all_normal = (how_many_true == len(normals)) #and is_all_normal
            # We always use a Mann Whitney U test.
            print(f"{metric_name_to_use:<40} normality test {'PASSED' if is_all_normal else' FAILED'}. Ones that Failed = {[n for n in normals if not n[0]]}")
            # We always use the Mann-Whitney U-test, since for MOST metrics, some methods failed normality.
            is_all_normal = False
            if is_all_normal:
                assert False
                test_str = f"We use a Welch T-test, since the data passed a Shapiro-Wilks normality test with $p = 0.05$. "
            else:
                test_str = f"We use a Mann-Whitney U-test, since the data failed a Shapiro-Wilks normality test with $p = 0.05$. "
            for method_name, dic_of_metrics in dic_of_all_values.items():
                this_metric_values = dic_of_metrics.get(
                    metric_name_to_use, [-1] * 5)
                if method_name == DEFAULT_MONIKER:
                    u, p, d, t = [-1] * 4
                else:
                    if is_all_normal:
                        t, p = ttest_ind(metric_values_to_compare_against,
                                         this_metric_values, alternative=alternative, equal_var=False)
                        u = -1
                    else:
                        u, p = mannwhitneyu(
                            metric_values_to_compare_against, this_metric_values, alternative=alternative, method='auto')
                        t = -1

                    d = (np.mean(metric_values_to_compare_against) -
                         np.mean(this_metric_values)) / (np.std(metric_values_to_compare_against))

                data[method_name] = ({'u': u, 't': t, 'p': p, 'd': d, 'mean': np.mean(
                    this_metric_values), 'std': np.std(this_metric_values)})

            optimal_mean = (min if 'time' in metric_name_to_use.lower() else max)([
                v['mean'] for v in data.values()
            ])
            for name, dic_of_stats_values in data.items():
                N = game + ' ' + metric_name_to_use
                dic_for_df[name][N] = mean_std(dic_of_stats_values, decimals=3 if metric_name_to_use == 'CompressionDistanceMetric' and game == 'Maze' else 2)
                p, d = dic_of_stats_values['p'], dic_of_stats_values['d']

                dic_for_df[name][N] = add_significance_to_dic_item(
                    dic_for_df[name][N], p, d, add_emblems=name != DEFAULT_MONIKER, is_highest= optimal_mean == dic_of_stats_values['mean'] and name_to_save_as != 'leniency')

    keys = list(dic_for_df[DEFAULT_MONIKER].keys())
    for metric_name in keys:
        new_metric_name = clean_metric(metric_name)
        if metric_name == new_metric_name:
            continue
        for name in dic_for_df:
            if metric_name in dic_for_df[name]:
                dic_for_df[name][new_metric_name] = dic_for_df[name][metric_name]
                del dic_for_df[name][metric_name]
    
    # Uses multi indexes
    new_better_dic_for_df = {}
    for name in dic_for_df:
        new_better_dic_for_df[name] = {}
        for metric_name, v in dic_for_df[name].items():
            splits = metric_name.split(" ")
            t = splits[0], " ".join(splits[1:])
            if 'Time' in t[1]:
                t = (t[0] + " Time (s)", t[1].replace("Time (s)", ""))
            new_better_dic_for_df[name][t] = v
    do_transpose = True
    if do_transpose:
        df = pd.DataFrame(new_better_dic_for_df)
    else:
        df = pd.DataFrame(dic_for_df)
    df = df.fillna('-')
    if do_transpose:
        df = df.T
    df = order_columns_of_df(df, rows=do_transpose)
    print(df)
    fpath = os.path.join(DIR_TO_SAVE, f'{name_to_save_as}.tex')
    buf = df.to_latex(None, escape=False,
                        sparsify=True,
                    #   column_format='l' + 'c' * (len(df.columns)),
                      column_format='l' + 'l' * (len(df.columns)),
                      multicolumn_format='c',
                      multicolumn=True
                      )
    n_rows_to_use = len(df) // 2
    lines = buf.split('\n')
    if not do_transpose:
        lines = lines[:n_rows_to_use+4] + [r"\midrule"] + lines[n_rows_to_use+4:]
    else:
        # do multi row things better
        splits = [l.strip().replace(r"\\", "") for l in lines[3].split('&')]
        if name_to_save_as == 'cd':
            splits_second_line = [s.strip().split(" ")[-1] for s in splits]
            splits = [' '.join(s.strip().split(" ")[:-1]) for s in splits]
        lines[3] = ' & '.join([r"\multicolumn{1}{c}{" + s + "}" if s != "{}" else s for s in splits ]) + r"\\"
        if name_to_save_as == 'cd':
            lines.insert(4, 
            ' & '.join([r"\multicolumn{1}{c}{" + s + "}" if s != "{}" else s for s in splits_second_line ]) + r"\\"
            )
        if name_to_save_as == 'solvability':
            splits = [l.strip().replace(r"\\", "") for l in lines[2].split('&')]
            lines[2] = ' & '.join([r"\multicolumn{1}{c}{" + s + " Solvability}" if s != "{}" else s for s in splits ]) + r"\\"
            lines.pop(3)
        # {} &                           Leniency &                     A* Difficulty &                          Leniency &                      A* Difficulty \\
    with open(fpath, 'w+') as f:
        f.writelines([l + "\n" for l in lines])

    with open(os.path.join(DIR_TO_SAVE, f'{name_to_save_as}_tests.tex'), 'w+') as f:
        f.write(test_str)


def v401_generation_time():
    """
        This compares the generation time of our method to that of our baselines
        under the null hypothesis that our method is comparable or slower than the others.

        We use a mann whitney u test, since the data showed non-normalities.
    """
    general_thing('generation_time', [
                  'Generation Time (s)', 'Train Time (s)'], alternative=['less', 'two-sided'])


def v402_solvability():
    """
        This compares the solvability of our method to that of our baselines
        under the null hypothesis that our method is comparable to the others.
    """
    general_thing('solvability', [
                  'SolvabilityMetric'], alternative='two-sided')


def v403_diversity_compression_distance():
    """
        This compares the solvability of our method to that of our baselines
        under the null hypothesis that our method is comparable to the others.
    """
    general_thing('cd', ['CompressionDistanceMetric', 
                        #  'AStarDiversityMetric',
                         'AStarEditDistanceDiversityMetric'], 
                         alternative='two-sided')


def v404_leniency_difficulty():
    """
        This compares the difficulty of our method to that of our baselines
        under the null hypothesis that our method is comparable to the others.
    """
    general_thing('leniency', ['LeniencyMetric',
                  'AStarDifficultyMetric'], alternative='two-sided')


def example_levels():
    """
    This should generate some example levels, specifically for:
        - Maze, for_maze_1 
            - For all methods
        - Mario, for_mario_1
            - For all methods
        - Maze Larger levels
            - One or two methods
        - Mario Larger levels
            _ One or two.
    """
    dir = 'results/v400/examples'

    def save_levels(dic_of_levels: Dict[str, List[Level]], name, is_mario=False):
        for method, levels in dic_of_levels.items():
            D = os.path.join(dir, name, method)
            os.makedirs(D, exist_ok=True)
            print(method, len(levels))
            for i, l in enumerate(levels):
                plt.figure(figsize=(40 * 2, 16 * 2))
                if is_mario:
                    plt.imshow(l.show(False))
                else:
                    plt.imshow(1 - l.map, vmin=0, vmax=1, cmap='gray')
                # plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"{D}/{i}.png", pad_inches=0.1,bbox_inches='tight')
                plt.close()

    save_levels(get_all_results_from_methods_mario(
        True)[-1], 'mario_normal', True)
    save_levels(get_all_results_from_methods_maze(
        True)[-1], 'maze_normal', False)



    # save a few super simple levels to illustrate the games.
    np.random.seed(42)
    map = np.random.rand(14, 14) > 0.5
    i = np.arange(len(map))
    map[i, (i+1) % 14] = 0
    map[(i-1), (i) % 14] = 0
    map[(i), (i) % 14] = 0
    plt.xticks([]); plt.yticks([]); plt.tight_layout()
    plt.imshow(1 - map, cmap='gray'); 
    plt.savefig(os.path.join(dir, 'maze_eg.png'), pad_inches=0.1,bbox_inches='tight')
    plt.close()

    mario_level = MarioLevel()
    mario_level.map[-1, 4:7] = 0
    mario_level.map[-2, 8] = 2
    mario_level.map[-4, 12] = 4
    mario_level.map[-5, 12] = 5
    
    mario_level.map[-2, 20] = 6
    mario_level.map[-3, 20] = 6
    plt.figure(figsize=(20, 6))
    plt.xticks([]); plt.yticks([]); plt.tight_layout()
    plt.imshow(mario_level.show(False)); 
    plt.savefig(os.path.join(dir, 'mario_eg.png'), pad_inches=0.1,bbox_inches='tight')
    plt.close()


def v410_presentation_plots(name_to_save_as: str, metric_name_to_use: str, 
                            message: str = '',
                            do_boxplot: bool = False):
    import seaborn as sns
    sns.set_theme()
    sns.set(font_scale = 1)

    # Can we show the same results in the tables, in plots, a bit more properly?
    # Fundamentally we'll be doing the same thing as metrics.py does, except with only one
    # level size.
    bad_names = ['DirectGA', 'DirectGA (Novelty)', 'PCGRL (Turtle)']
    palette = None
    for num, game in (enumerate(['Maze', "Mario"])):
        if game == 'Maze':
            dic_of_mean_stds, dic_of_all_values, dic_all_results = get_all_results_from_methods_maze(return_all_of_the_results_not_just_single=True)
        else:
            dic_of_mean_stds, dic_of_all_values, dic_all_results = get_all_results_from_methods_mario(return_all_of_the_results_not_just_single=True)
        
        if palette is None:
            keys_to_consider = set(dic_all_results.keys()) - set(bad_names)
            keys_to_consider = sorted(keys_to_consider)
            I = keys_to_consider.index(DEFAULT_MONIKER)
            keys_to_consider = [DEFAULT_MONIKER] + keys_to_consider[:I] + keys_to_consider[I+1:]

            palette = dict(zip(keys_to_consider, sns.color_palette(n_colors=len(keys_to_consider))))
        print(dic_all_results[DEFAULT_MONIKER].keys())
        M = clean_metric(metric_name_to_use)

        big_dic_of_all_methods = {'Method': [], M: []}
        legend = []
        # for moniker, list_of_5_seeds in dic_all_results.items():
        for moniker in keys_to_consider:
            list_of_5_seeds = dic_all_results[moniker]
            if moniker in bad_names: continue
            # 5 seeds, so need to list sum this
            if type(list_of_5_seeds[metric_name_to_use][0]) == float or type(list_of_5_seeds[metric_name_to_use][0]) == int:
                list_of_5_seeds[metric_name_to_use] =  [[i] for i in list_of_5_seeds[metric_name_to_use]]
            ARR = sum(map(list, list_of_5_seeds[metric_name_to_use]), [])
            
            big_dic_of_all_methods['Method'].extend([moniker] * len(ARR))
            big_dic_of_all_methods[M].extend(ARR)
            legend.append(moniker)
        if do_boxplot:
            sns.boxplot(x="Method", y=M, data=big_dic_of_all_methods, palette=palette)
            plt.yscale('log')
            plt.ylabel('Time (s) - Lower is Better', fontsize=20)
        else:
            g = sns.histplot(big_dic_of_all_methods, x=M, hue='Method', palette=palette)
            # , legend=True
            # g.legend_.set_title(None)
        
        # plt.legend()
        plt.title(f"{game}: {M}", fontsize=20)
        if len(message):
            plt.xlabel(message)
        DIR = 'results/v400/presentation_plots'
        os.makedirs(DIR, exist_ok=True)
        # plt.savefig(os.path.join(DIR, game + "_" + metric_name_to_use + ".png"), bbox_inches='tight', pad_inches=0.1)
        plt.savefig(os.path.join(DIR, game + "_" + metric_name_to_use + ".png"))
        plt.close()


def v420_exp_108():
    import seaborn as sns
    sns.set_theme()

    # This goes over experiment 108, showing the generation time as a function of 
    # num generations for NEAT and DirectGA on Maze.
    neat_path = '../results/experiments/experiment_108_a/Maze/NEAT'
    directga_path = '../results/experiments/experiment_108_a/Maze/DirectGA'
    def get_things(name):
        files = glob.glob(os.path.join(name, '*', '*', '**', '*.p'), recursive=True)
        dic_of_gens_to_time = defaultdict(lambda: [])
        for f in files:
            num_gens = int(f.split("/")[8])
            with open(f, 'rb') as file:
                d = pickle.load(file)
                dic_of_gens_to_time[num_gens].append(d['generation_time'])

        return dic_of_gens_to_time
        pass
    neat_results = get_things(neat_path)
    direct_results = get_things(directga_path)
    all_gens = sorted(list(neat_results.keys()))
    def plot_thing(dic, name):
        list_of_means = []
        list_of_stds = []
        for g in all_gens:
            list_of_means.append(np.mean(dic[g]))
            list_of_stds.append(np.std(dic[g]))
        
        list_of_means = np.array(list_of_means) 
        list_of_stds = np.array(list_of_stds) 

        plt.plot(all_gens, list_of_means, label=name)
        plt.fill_between(all_gens, list_of_means - list_of_stds, list_of_means + list_of_stds, alpha=0.5)
    
    plot_thing(neat_results, 'NEAT')
    plot_thing(direct_results, 'DirectGA+')
    plt.xlabel("Number of generations")
    plt.ylabel("Generation Time (s)")
    plt.yscale('log')
    plt.title("Number of generations vs generation time - Maze")
    plt.legend()
    dir = 'results/v400/num_gens_effect'
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, 'v108.png')) #  , pad_inches=0.1,bbox_inches='tight'
    plt.close()
            
def more_presentation():
    # Plots some levels to illustrate stuff.
    W = 10
    DIR = 'results/v400/presentation_plots/levels'; os.makedirs(DIR, exist_ok=True)
    def solvs():
        np.random.seed(42)
        map = np.random.rand(W, W) > 0.5
        map[1, 1:3] = 0
        def do_plot(map, name, traj=None):
            path, sets, num_considered = do_astar_from_level(MazeLevel.from_map(map))
            plt.imshow(1 - map, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.gca().set_position((0, 0, 1, 1))
            if path is not None:
                plt.plot(*zip(*path), linewidth=5, color='green')
                # plt.scatter([path[-1][0]], path[-1][1], marker='$\checkmark$', color='green', s=100, linewidths=4, alpha=1, zorder=1000)
            if traj is not None:
                plt.plot(*zip(*traj), linewidth=5, color='green')
                plt.scatter([traj[-1][0]], traj[-1][1], marker='x', color='red', s=100, linewidths=5, alpha=1, zorder=1000)

            plt.savefig(os.path.join(DIR, name), bbox_inches='tight', pad_inches=0)
            plt.close()
        do_plot(map, 'solv-1.png')
        map = np.random.rand(W, W) > 0.5
        map[2, 1] = 0
        do_plot(map, 'solv-2.png', traj=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (5, 4), (5, 5)])
        
        map = np.random.rand(W, W) > 0.5; map = np.random.rand(W, W) > 0.5
        map[2, 2] = 0
        map[6, 4] = 0
        map[7, 6] = 0
        map[-1, -2:] = 0
        do_plot(map, 'solv-3.png')
    
    def novelty():
        np.random.seed(42)
        map = np.random.rand(W, W) > 0.5
        map[1, 1:3] = 0
        def do_plot(map, name, traj=None):
            path, sets, num_considered = do_astar_from_level(MazeLevel.from_map(map))
            plt.imshow(1 - map, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.gca().set_position((0, 0, 1, 1))
            plt.savefig(os.path.join(DIR, name), bbox_inches='tight', pad_inches=0)
            plt.close()
        do_plot(map, 'nov-1.png')
        map[4, 4] = 1
        _ = np.random.rand(W, W) > 0.5
        do_plot(map, 'nov-2.png')
        
        map = np.random.rand(W, W) > 0.5; map = np.random.rand(W, W) > 0.5
        map[2, 2] = 0
        map[6, 4] = 0
        map[7, 6] = 0
        map[-1, -2:] = 0
        do_plot(map, 'nov-3.png')

        map = np.random.rand(W, W) > 0.5
        # other ones
        map[1, 1:3] = 0
        map[4, 4] = 0
        do_plot(map, 'nov-4.png')
        map = np.random.rand(W, W) > 0.5
        map[:, 0] = 0
        map[-1, :] = 0
        do_plot(map, 'nov-5.png')
        
        map = np.random.rand(W, W) > 0.5; map = np.random.rand(W, W) > 0.5
        map = np.random.rand(W, W) > 0.5; map = np.random.rand(W, W) > 0.5
        map[:, -1] = 0
        map[0, :] = 0
        do_plot(map, 'nov-6.png')
    solvs()
    novelty()

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    v410_presentation_plots('', 'AStarEditDistanceDiversityMetric', message='Diversity - How different are levels from the same method');
    more_presentation();
    v420_exp_108();
    example_levels();
    v401_generation_time()
    v402_solvability();
    v403_diversity_compression_distance()
    v404_leniency_difficulty()

