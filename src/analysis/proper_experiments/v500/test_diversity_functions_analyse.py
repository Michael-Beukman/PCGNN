from collections import defaultdict
import glob
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import ray
from analysis.proper_experiments.v400.analyse_all_statistical_tests import clean_metric, mean_std
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel

from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel

from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarEditDistanceDiversityMetric, AStarSolvabilityMetric
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.metric import Metric
from matplotlib import pyplot as plt
from common.utils import mysavefig

"""This analyses the novelty distance function's results results
"""

def get_results(dic_of_all_methods):
    def get_metrics() -> Tuple[List[Metric], Metric]:
        game = MazeGame(MazeLevel())
        parent = AStarDiversityAndDifficultyMetric(game, number_of_times_to_do_evaluation=5)
        return [
            AStarSolvabilityMetric(game, parent),
            CompressionDistanceMetric(game),
            LeniencyMetric(game),
            AStarDiversityMetric(game, parent),
            AStarDifficultyMetric(game, parent),
            AStarEditDistanceDiversityMetric(game, parent),
        ], parent
    
    @ray.remote
    def single(dic, name, I):
        metrics, _parent = get_metrics()
        answer = {}
        if 1:
            levels = dic['levels']
            if 'generation_time' in dic:
                gentime = dic.get('generation_time', -1)
            else:
                gentime = dic.get('time_per_level', -2)
            traintime = dic.get('train_time', -3)

            answer['generation_time'] = gentime
            answer['train_time'] = traintime

        for m in metrics:
            li_of_metrics = np.array(m.evaluate(levels))
            answer[m.name()] = list(li_of_metrics)
            print(f"Name {name}, I = {I}. Metric = {m.name()}; Value = {np.mean(answer[m.name()])}")
        answer['parent_metric'] = _parent
        return answer

    all_futures = []
    parents = []
    pickle_filenames = []
    for method_name in dic_of_all_methods.keys():
        list_of_vals = dic_of_all_methods[method_name]
        for i, V in enumerate(list_of_vals):
            all_futures.append(single.remote(V, method_name, i))
            parents.append(method_name)

    all_answers = ray.get(all_futures)
    dic = {
        k: {
            'names': [],
            'results_single': defaultdict(lambda: []),
            'results_all': defaultdict(lambda: []),
            'parent_metrics': []
        } for k in set(parents)
    }
    for parent, ans in zip(parents, all_answers):
        for key, value in ans.items():
            if key == 'parent_metric':
                dic[parent]['parent_metrics'].append(value)
            else:
                dic[parent]['results_single'][key].append(np.mean(value))
                dic[parent]['results_all'][key].append(value)


    DIR = f'../results/v500/maze/results/batch'
    os.makedirs(DIR, exist_ok=True)
    file = os.path.join(DIR, 'data.p')
    # convert to normal dictionary because defaultdict breaks pickle :)
    for k in dic:
        dic[k]['results_single'] = dict(dic[k]['results_single'])
        dic[k]['results_all'] = dict(dic[k]['results_all'])
    
    dic = dict(dic)

    with open(file, 'wb+') as f:
        pickle.dump(dic, f)
    return file


def get_results_mario(dic_of_all_methods):
    ray.init()
    def get_metrics() -> Tuple[List[Metric], Metric]:
        game = MarioGame(MarioLevel())
        parent = AStarDiversityAndDifficultyMetric(game, number_of_times_to_do_evaluation = 5)
        return [
            AStarSolvabilityMetric(game, parent),
            CompressionDistanceMetric(game),
            CompressionDistanceMetric(game, use_combined_features=True),
            CompressionDistanceMetric(game, do_mario_flat=True),
            LeniencyMetric(game),
            AStarDiversityMetric(game, parent),
            AStarDifficultyMetric(game, parent),
            AStarEditDistanceDiversityMetric(game, parent),
        ], parent
    
    @ray.remote
    def single(dic, name, I):
        metrics, _parent = get_metrics()
        answer = {}
        if 1:
            levels = dic['levels']
            if 'generation_time' in dic:
                gentime = dic.get('generation_time', -1)
            else:
                gentime = dic.get('time_per_level', -2)
            traintime = dic.get('train_time', -3)

            answer['generation_time'] = gentime
            answer['train_time'] = traintime

        for m in metrics:
            li_of_metrics = np.array(m.evaluate(levels))
            answer[m.name()] = list(li_of_metrics)
            print(f"Name {name}, I = {I}. Metric = {m.name()}; Value = {np.mean(answer[m.name()])}")
        answer['parent_metric'] = _parent
        return answer

    all_futures = []
    parents = []
    pickle_filenames = []
    for method_name in dic_of_all_methods.keys():
        list_of_vals = dic_of_all_methods[method_name]
        for i, V in enumerate(list_of_vals):
            all_futures.append(single.remote(V, method_name, i))
            parents.append(method_name)

    all_answers = ray.get(all_futures)
    dic = {
        k: {
            'names': [],
            'results_single': defaultdict(lambda: []),
            'results_all': defaultdict(lambda: []),
            'parent_metrics': []
        } for k in set(parents)
    }
    for parent, ans in zip(parents, all_answers):
        for key, value in ans.items():
            if key == 'parent_metric':
                dic[parent]['parent_metrics'].append(value)
            else:
                dic[parent]['results_single'][key].append(np.mean(value))
                dic[parent]['results_all'][key].append(value)


    DIR = f'../results/v500/mario/results/batch/'
    os.makedirs(DIR, exist_ok=True)
    file = os.path.join(DIR, 'data.p')
    # convert to normal dictionary because defaultdict breaks pickle :)
    for k in dic:
        dic[k]['results_single'] = dict(dic[k]['results_single'])
        dic[k]['results_all'] = dict(dic[k]['results_all'])
    
    dic = dict(dic)

    with open(file, 'wb+') as f:
        pickle.dump(dic, f)
    return file


def bold_extreme_values(data, series, format_string="%.2f", max_=True):
    means = series.map(lambda x: float(x.split(" ")[0].replace("$", "")))
    maxes = means != means.max()
    mins = means != means.min()
    
    red = series.apply(lambda x : "\\textcolor{darkgreen}{%s}" % x)
    blue = series.apply(lambda x : "\\textcolor{blue}{%s}" % x)
    formatted = series.apply(lambda x : x)
    return formatted.where(maxes, blue).where(mins, red)


def save_thing(df, game):
    for col in df.columns.get_level_values(0).unique():
        df[col] = bold_extreme_values(None, df[col], max_=True)
    
    DIR_TO_SAVE = 'results/v500/distance/'
    os.makedirs(DIR_TO_SAVE, exist_ok=True)
    fpath = os.path.join(DIR_TO_SAVE, f'{game}.tex')
    buf = df.to_latex(None, escape=False,
                        sparsify=True,
                      column_format='l' + 'l' * (len(df.columns)),
                      multicolumn_format='c',
                      multicolumn=True
                      )
    lines = buf.split('\n')
    with open(fpath, 'w+') as f:
        f.writelines([l + "\n" for l in lines])


def get_example_levels(folder, seed_use =0, i = 0, name='maze'):
    F = f'../results/v500/{name}/results/levels'
    os.makedirs(F, exist_ok=True)
    A = glob.glob(folder, recursive=True)
    dic = {}
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    for method_idx, file in enumerate(A):
        name = file.split("/")[-1]
        dic[name] = []
        for seed_file in sorted(glob.glob(os.path.join(file, f'{seed_use}/*.p'))):
            with open(seed_file, 'rb') as f:
                levels = pickle.load(f)['levels']
                levels[i].show(True)
                plt.xticks([]); plt.yticks([])
                plt.title(clean_name_of_distance(name), fontsize=18)
                plt.tight_layout()
                mysavefig(os.path.join(F, f'{name}_{seed_use}_{i}.png'), bbox_inches='tight', pad_inches=0.01)
                plt.close()
                
def clean_name_of_distance(n):
    D ={
    'dist_compare_shortest_paths': 'Path',
    'rolling_window_comparison_what_you_see_from_normal_default': 'Window',
    'jensen_shannon_compare_trajectories_distance': 'JS',
    'visual_diversity_only_reachable': 'Visual Diversity Reachable',
    'image_hash_distance_wavelet': 'Hashing (Wavelet)',
    'euclidean_distance': 'Euclidean',
    'image_hash_distance_perceptual': 'Hashing (Perceptual)',
    'rolling_window_comparison_what_you_see_from_normal_default_TRAJ': 'Window (V2)',
    'visual_diversity': 'Visual Diversity',
    'visual_diversity_normalised': 'Visual Diversity',
    'image_hash_distance_average': 'Hashing (Average)',
    'image_hash_distance_perceptual_simple': 'Hashing (Perceptual Simple)',}
    if n in D: return D[n]
    return D['_'.join(n.split("_")[:-1])]

            
def sort_df(df, is_maze):
    a = [
'Path',
'Window (V2)',
'JS',
'Visual Diversity Reachable',
'Hashing (Wavelet)',
'Euclidean',
'Hashing (Perceptual)',
'Window',
'Visual Diversity',
'Hashing (Average)',
'Hashing (Perceptual Simple)',
    ]  if is_maze else [
'Hashing (Perceptual)',
'Visual Diversity',
'Hashing (Average)',
'Hashing (Wavelet)',
'Hashing (Perceptual Simple)',
'Euclidean',
    ]      
    a = [(i, s) for i, s in enumerate(a) if s in df.index]


    ordering = a
    if True:
        print(df.index)
        return df.reindex([o[1] for o in ordering]) 
def read_analyse_results(
    MAZE = False
):
    
    if MAZE:
        FILE = '../results/v500/maze/results/batch/data.p'
        NAME = 'Maze'
    else:
        FILE = '../results/v500/mario/results/batch/data.p'
        NAME = 'Mario'
    
    with open(FILE, 'rb') as f:
        dic = pickle.load(f)

        ans = {}
        metrics_to_check = ['generation_time', 'train_time', 'SolvabilityMetric', 'CompressionDistanceMetric', 'AStarEditDistanceDiversityMetric', 'LeniencyMetric', 'AStarDifficultyMetric']
        metric = 'SolvabilityMetric'
        for metric in metrics_to_check:
            for name in dic:
                if not MAZE:
                    dic[name]['results_single']['CompressionDistanceMetric'] = dic[name]['results_single']['CompressionDistanceMetric_CombinedFeatures']
                VV = []
                for i in range(5):
                    VV.append(dic[name]['results_single'][metric][i])
                mean, std = np.mean(VV), np.std(VV)
                
                SS = mean_std(dict(mean=mean, std=std),decimals=3 if metric == 'CompressionDistanceMetric' and MAZE else 2)
                MM = np.round(mean, 2)
                if MM == 0 and mean != 0:
                    SS = f'{np.round(mean, 3)} ({np.round(std, 1)})'
                
                if MM > 500:
                    SS = f'{int(np.round(mean))} ({int(np.round(std))})'
                
                M = clean_metric(metric)
                
                NN = clean_name_of_distance(name)
                if NN not in ans:
                    ans[NN] = {M: SS}
                else:
                    ans[NN][M] = SS
        df = pd.DataFrame(ans).T
        df.index.name = ('Distance Function')
        df = sort_df(df, MAZE)
        save_thing(df, NAME)
        

def main():
    read_analyse_results(False)
    read_analyse_results(True)
    # Plot Levels
    for seed in range(5):
        for i in range(5):
            get_example_levels('../results/experiments/experiment_501_a/Maze/NEAT/batch/*/*', i=i);
            get_example_levels('../results/experiments/502_a/Mario/NeatNovelty/batch/*/*', i=i, seed_use=seed, name='mario');

if __name__ == '__main__':
    eval_mario = True
    eval_maze = True
    if eval_mario:
        A = glob.glob('../results/experiments/502_a/Mario/NeatNovelty/batch/*/*', recursive=True)
        dic = {}
        for file in A:
            name = "_".join(file.split("/")[-1].split("_")[:-1])
            dic[name] = []
            for seed_file in sorted(glob.glob(os.path.join(file, '*/*.p'))):
                with open(seed_file, 'rb') as f:
                    dic[name].append(pickle.load(f))
        get_results_mario(dic);
    
    if eval_maze:
        A = glob.glob('../results/experiments/experiment_501_a/Maze/NEAT/batch/*/*', recursive=True)
        dic = {}
        for file in A:
            name = file.split("/")[-1]
            dic[name] = []
            for seed_file in sorted(glob.glob(os.path.join(file, '*/*.p'))):
                with open(seed_file, 'rb') as f:
                    dic[name].append(pickle.load(f))
        get_results(dic); 
        
        
    main()
