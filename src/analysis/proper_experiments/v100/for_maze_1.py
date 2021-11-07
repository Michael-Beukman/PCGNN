"""
    This file is specifically for taking a bunch of levels that were generated for the maze game, 
    and evaluating them using the latest version of the metrics.

    It stores the results to `results/maze/metrics_and_levels/`
"""

from collections import defaultdict
import glob
import os
import pickle
import pprint
from typing import Any, Dict, List, Tuple
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import ray
from common.utils import get_date, get_latest_folder, get_only_solvable_metrics
from experiments.config import Config

from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarEditDistanceDiversityMetric, AStarSolvabilityMetric
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.metric import Metric

def pretty_key(k):
    if 'time' in k:
        return ' '.join(map(str.title, k.split("_"))) + " (s)"
    return k

def make_nice(s):
    # Cleans up names
    if 'neat_best_0815' in s:
        return 'NoveltyNEAT (Ours)'
    K =  ' '.join(map(str.title, s.split("_")))
    K = K.replace("Directga", "DirectGA")
    K = K.replace("Withnovelty", "With Novelty")
    K = K.replace("Perfectsolve", "")
    return K

def get_astar_metrics(one_run: Dict[str, Any], do_results_mean_std=False):
    levels = one_run['levels']
    game = MazeGame(MazeLevel())
    parent = AStarDiversityAndDifficultyMetric(game, number_of_times_to_do_evaluation=5)

    div = AStarDiversityMetric(game, parent)
    diff = AStarDifficultyMetric(game, parent)
    
    val_div = div.evaluate(levels)
    val_diff = diff.evaluate(levels)
    
    one_run['eval_results_single'][div.name()] = np.mean(val_div)
    one_run['eval_results_single'][diff.name()] = np.mean(val_diff)


    one_run['eval_results_all'][div.name()] = val_div
    one_run['eval_results_all'][diff.name()] = val_diff

    one_run = clean_the_single_results_of_one_run(one_run)
    return one_run


def clean_the_single_results_of_one_run(one_run):
    # Cleans results and only shows solvable levels.
    solvs = one_run['eval_results_all']['SolvabilityMetric']
    for key, li in one_run['eval_results_all'].items():
        if key == 'SolvabilityMetric': continue
        print(f"KEY = {key}, lengths = {len(li)}")
        if len(li) == 1:
            continue
        new = np.mean(get_only_solvable_metrics(li, solvs))
        old = one_run['eval_results_single'][key]
        one_run['eval_results_single'][key] = new

        print(f"{key:<25} | OLD = {np.round(old, 3)} | NEW = {np.round(new, 3)}")
        
    return one_run


def rerun_all_of_the_metrics():    
    """
    This takes the results from a few different runs
    and displays them properly.
    """
    ray.init()
    all_main_dirs = glob.glob(os.path.join(get_latest_folder('../results/experiments/experiment_102_aaa_rerun_only_best/Maze/DirectGA/2*'), '*/*/*/*/*/')) + \
                    glob.glob(get_latest_folder("../results/experiments/experiment_105_a/Maze/NEAT/2*/50/200/")) + \
                    glob.glob(get_latest_folder("../results/experiments/experiment_102_f_visual_diversity_rerun_batch/Maze/DirectGA/2*/50/100/0/True/True/")) + \
                    glob.glob('../results/all_pcgrl/binary/turtle/100000000.0/inferred_levels/') + \
                    glob.glob('../results/all_pcgrl/binary/wide/100000000.0/inferred_levels/')
    
    organised_files = {}
    organised_files = dict(organised_files)
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
    def single(pickle_name, I):
        metrics, _parent = get_metrics()
        answer = {}
        with open(pickle_name, 'rb') as f:
            dic = pickle.load(f)
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
            print(f"Name {pickle_name}, I = {I}. Metric = {m.name()}; Value = {np.mean(answer[m.name()])}")
        answer['parent_metric'] = _parent
        return answer

    all_futures = []
    parents = []
    pickle_filenames = []
    for m in all_main_dirs:
        globs = glob.glob(os.path.join(m, '*/*.p'))
        if 'all_pcgrl' in m:
            globs = glob.glob(os.path.join(m, 'run_seed*.p'))
        globs = sorted(globs)
        print(f"For Len = {len(globs)}")#M = {m}, . globs = {globs}")
        organised_files[m] = globs
        all_futures += [single.remote(f, i) for i, f in enumerate(globs)]
        for i in range(len(globs)): parents.append(m)
        pickle_filenames.extend(globs)
    
    all_answers = ray.get(all_futures)
    dic = {
        k: {
            'names': [],
            'results_single': defaultdict(lambda: []),
            'results_all': defaultdict(lambda: []),
            'parent_metrics': []
        } for k in set(parents)
    }
    for parent, name, ans in zip(parents, pickle_filenames, all_answers):
        dic[parent]['names'].append(name)
        for key, value in ans.items():
            if key == 'parent_metric':
                dic[parent]['parent_metrics'].append(value)
            else:
                dic[parent]['results_single'][key].append(np.mean(value))
                dic[parent]['results_all'][key].append(value)


    DIR = f'results/maze/metrics_and_levels/{get_date()}'
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


if __name__ == '__main__':
    rerun_all_of_the_metrics()