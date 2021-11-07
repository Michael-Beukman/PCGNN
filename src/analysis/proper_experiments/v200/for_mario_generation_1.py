from collections import defaultdict
from email.policy import default
import glob
import hashlib
import os
import pickle
from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import ray
import scipy
from common.utils import bold_pandas_dataframe, do_statistical_tests_and_get_df, get_date, get_latest_folder, get_only_solvable_metrics
from games.mario.mario_game import MarioGame

from games.mario.mario_level import MarioLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarEditDistanceDiversityMetric, AStarSolvabilityMetric
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.metric import Metric
from metrics.solvability import SolvabilityMetric
from scipy.stats import wilcoxon, ttest_ind, friedmanchisquare, mannwhitneyu
from scipy.stats.distributions import chi2 as CHI2_Dist

def main():
    """
    This tries to do the following:
    - For our top few methods, i.e
        - MarioIsland
        - MarioNeat single pop
        - Mario Ferreira GA

    Possibly rerun the metrics, and run the A* metrics properly

    Then analyse them and create a table like metrics.tex for the maze case.
    """
    ray.init(num_cpus=6)
    filenames = [

        # 204e, best parameters.
        get_latest_folder('../results/experiments/experiment_204e/Mario/NeatNovelty/*/100/150/True/1/0/6/2/-1/True/'),

        # 201 exp, default params
        get_latest_folder('../results/experiments/experiment_201_b/Mario/DirectGA/*/20/100/0.0/0.0/1.0/0.5/114/20/10/10/2/2'),

        # 201 exp, best params
        get_latest_folder('../results/experiments/experiment_201_a/Mario/DirectGA/*/10/50/0.5/0.5/0.5/0.5/20/20/10/40/2/2'),

        # Mario Novelty, DirectGA
        get_latest_folder('../results/experiments/experiment_201_d/Mario/DirectGA/*/100/100/0.0/0.0/1.0/0.5/114/20/10/10/2/2/use_novelty'),
        
        # PCGRL.
        '../results/all_pcgrl/smb/turtle/100000000.0/inferred_levels_v2',
        '../results/all_pcgrl/smb/wide/100000000.0/inferred_levels_v2/',
    ]

    def get_metrics() -> Tuple[List[Metric], Metric]:
        game = MarioGame(MarioLevel())
        parent = AStarDiversityAndDifficultyMetric(game, number_of_times_to_do_evaluation = 5)
        return [
            # SolvabilityMetric(game),
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
    def single(pickle_name, I):
        metrics, _parent = get_metrics()
        with open(pickle_name, 'rb') as f:
            dic = pickle.load(f)
            levels = dic['levels']
        
        answer = {}
        for m in metrics:
            li_of_metrics = np.array(m.evaluate(levels))
            answer[m.name()] = list(li_of_metrics)
            print(f"Name {pickle_name}, I = {I}. Metric = {m.name()}; Value = {np.mean(answer[m.name()])}")
        answer['parent_metric'] = _parent
        return answer
    
    all_futures_overall = []
    pickle_filenames = []
    parents = []
    I = 0
    for parent in filenames:
        globs = glob.glob(os.path.join(parent, '*/*.p'))
        if 'pcgrl' in parent:
            # different folder structure.
            globs = glob.glob(os.path.join(parent, '*.p'))
        globs = sorted(globs)
        for x in globs:
            all_futures_overall.append(single.remote(x, I))
            parents.append(parent)
            pickle_filenames.append(x)
            I += 1
    
    print("GETTING ", len(all_futures_overall), "Runs")
    all_answers = ray.get(all_futures_overall)
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


    DIR = f'results/mario/metrics_and_levels/{get_date()}'
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

def analyse(MY_FILE):

    FILES = [
        # multiple times solvability.
        MY_FILE
    ]
            
    overall_dic = {}
    def better(v):
        m, s = v['mean'], v['std']
        m = np.round(m, 2)
        s = np.round(s, 2)
        return f"{m} ({s})", v['all']
    
    def get_moniker(name):
        if 'v14' in name: return "NEAT Island"
        if 'v10' in name: return "NEAT Single Population"
        if 'experiment_204e' in name: return "NEAT (204e)"
        
        if '201_a' in name: return "Optimised DirectGA"
        if '201_b' in name: return "Default DirectGA"
        if 'pcgrl' in name and 'wide' in name: return "PCGRL (Wide)"

    def prett(metric_name):
        return ' '.join(metric_name.split("_"))
    all_dics = {}
    for FILE in  FILES:
        with open(FILE, 'rb') as f:
            all_dics |= pickle.load(f)
    dic = all_dics
    overall_dic_for_stats_test = {}
    if 1:
        # dic = pickle.load(f)
        for single_run in dic:
            moniker = get_moniker(single_run)
            print(single_run, moniker)
            little_d = {}; stats_d = {}
            dic[single_run] = clean_the_single_results_of_one_run(dic[single_run])
            single_results = dic[single_run]['results_single']

            names = dic[single_run]['names']
            gentimes, traintimes = [], []
            levels = []
            for n in names:
                with open(n, 'rb') as f:
                    D = pickle.load(f)
                    
                    gen_time = D.get('generation_time', D.get('time_per_level'))
                    train_time = D.get('train_time', -1)
                    gentimes.append(gen_time)
                    traintimes.append(train_time)
                    levels.append(D['levels'][0])
            dir_for_plots = os.path.join('results/mario/1_metrics', moniker.replace(' ', '_'))
            os.makedirs(dir_for_plots, exist_ok=True)
            for i, l in enumerate(levels):
                plt.figure(figsize=(20, 2.5))
                plt.imshow(l.show(False))
                plt.savefig(os.path.join(dir_for_plots, f'{i}.png'), pad_inches=0.1, bbox_inches='tight')
                plt.close()

            for key, li in single_results.items():
                mean, std = np.mean(li), np.std(li)
                if key == 'SolvabilityMetric' and 'neat' in moniker.lower():
                    print(f"SOLVABILITY: {li}, mean = {mean}, std = {std}")
                little_d[prett(key)], stats_d[prett(key)] = better({'mean': mean, 'std': std, 'all': li})
            little_d['Generation Time (s)'], stats_d['Generation Time (s)'] = better({'mean': np.mean(gentimes), 'std': np.std(gentimes), 'all': gentimes})
            little_d['Train Time (s)'], stats_d['Train Time (s)'] = better({'mean': np.mean(traintimes), 'std': np.std(traintimes), 'all': gentimes})

            overall_dic[moniker] = little_d
            overall_dic_for_stats_test[moniker] = stats_d
    
    print(overall_dic_for_stats_test)
    overall_dic_for_p_values = defaultdict(lambda: {})
    default_moniker_to_compare_against = "NEAT (204e)"  #'NEAT Single Population'

    df, df_stats = do_statistical_tests_and_get_df(overall_dic, overall_dic_for_stats_test, default_moniker_to_compare_against)

    dir = 'results/mario/1_metrics'
    os.makedirs(dir, exist_ok=True)
    

    df.to_latex(f"{dir}/metrics.tex", escape=False)
    df_stats.to_latex(f"{dir}/metrics_stats.tex", escape=False)


def clean_the_single_results_of_one_run(one_run):
    # Cleans results and only shows solvable levels.
    for K in range(len(one_run['results_all']['SolvabilityMetric'])):
        solvs = one_run['results_all']['SolvabilityMetric'][K]
        for key, _ in one_run['results_all'].items():
            if key == 'SolvabilityMetric': continue
            li = one_run['results_all'][key][K]
            print(f"KEY = {key}, lengths = {len(li)}")
            if len(li) == 1:
                continue
            new = np.mean(get_only_solvable_metrics(li, solvs))
            old = one_run['results_single'][key][K]
            one_run['results_single'][key][K] = new

            print(f"{key:<25} | OLD = {np.round(old, 3)} | NEW = {np.round(new, 3)}")
            
    return one_run


if __name__ == '__main__':
    F = main()
    analyse(F)
