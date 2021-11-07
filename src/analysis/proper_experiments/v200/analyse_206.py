from collections import defaultdict
import glob
import os
import pickle
import re
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from common.utils import get_latest_folder, get_only_solvable_metrics

FILE_TO_USE = get_latest_folder('../results/experiments/206a/runs/*/data.p')
DIRECT_GA_PATH = get_latest_folder('../results/experiments/experiment_206b/Mario/DirectGA/2*')
DIRECT_GA_OPTIM_PATH = get_latest_folder('../results/experiments/experiment_206c/Mario/DirectGA/2*')

def main(FILE=FILE_TO_USE):
    """
    Analyses experiment 206
    """

    with open(FILE, 'rb') as f:
        dic = pickle.load(f)
        og = dic['original']
        print(dic.keys())
        D = dic['data']
        print(D.keys())
        for w in D:
            L = (D[w][0]['levels'])[0]
            print(f"W = {w}, level size = {L.map.shape}, w2 = {D[w][0]['w']}")


def pretty_key(k):
    if 'time' in k:
        return ' '.join(map(str.title, k.split("_"))) + " (s)"
    # Thanks :) https://stackoverflow.com/a/37697078
    splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', k)).split()
    return ' '.join(splitted)
    return ' '.join(map(str.title, k.split("_")))
    return k


def analyse_206_with_line_graph(FILE=FILE_TO_USE):
    """
        This plots a line graph of experiment 206
        Basically copied from analyse_104
    """
    with open(FILE, 'rb') as f:
        data = pickle.load(f)

    
    def get_mean_standard_for_one_point_in_directga(width, mean_dic, std_dic, which='normal'):
        if which == 'normal':
            path = f'{DIRECT_GA_PATH}/{width}/*/*/*/*.p'
        else:
            path = f'{DIRECT_GA_OPTIM_PATH}/{width}/*/*/*/*.p'

        li = glob.glob(path)
        assert len(li) == 5
        metrics = defaultdict(lambda: [])
        all_levels = []
        for p in li:
            with open(p, 'rb') as f:
                d = pickle.load(f)
                for key in d['eval_results_single']:
                    if key != 'SolvabilityMetric':
                        V = np.mean(get_only_solvable_metrics(d['eval_results_all'][key], d['eval_results_all']['SolvabilityMetric']))
                    else:
                        V = np.mean(d['eval_results_all'][key])
                    metrics[key].append(V)
                for key in ['generation_time']:
                    # DON'T divide by 100 here, as this was for 1 level. The experiment.py already normalised it.
                    metrics[key].append(d[key])
                all_levels.append(d['levels'][0])
        dir = f'results/mario/206/line_graph/levels_direct_ga_{which}'
        for i, l in enumerate(all_levels):
            os.makedirs(dir, exist_ok=True)
            plt.figure(figsize=(40, 16))
            plt.imshow(l.show(False))
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
    for i, f in enumerate(fs):
        with open(f, 'rb') as pickle_file:
            t = pickle.load(pickle_file)['generation_time']
            # D[114][i]['generation_time'] = t;
            og_metrics['generation_time'] +=t
    # Get og metric results
    for T in data['original']:
        things = T['eval_results_single']
        for key in things:
            if key != 'SolvabilityMetric':
                V = np.mean(get_only_solvable_metrics(T['eval_results_all'][key], T['eval_results_all']['SolvabilityMetric']))
            else:
                V = np.mean(T['eval_results_all'][key])
            # og_metrics[key] += np.mean(things[key])
            og_metrics[key] += np.mean(V)

    for key in og_metrics:
        og_metrics[key] /= len(fs)
    all_metrics = {
        14: og_metrics
    }
    all_values_mean = defaultdict(lambda : [])
    all_values_std = defaultdict(lambda : [])
    
    all_values_mean_direct_ga = defaultdict(lambda : [])
    all_values_std_direct_ga = defaultdict(lambda : [])

    directga_widths = [28, 56, 85, 114, 171, 228]
    for w in directga_widths: 
        get_mean_standard_for_one_point_in_directga(w, all_values_mean_direct_ga, all_values_std_direct_ga)


    
    all_values_mean_direct_ga_optim = defaultdict(lambda : [])
    all_values_std_direct_ga_optim = defaultdict(lambda : [])
    for w in directga_widths: 
        get_mean_standard_for_one_point_in_directga(w, all_values_mean_direct_ga_optim, all_values_std_direct_ga_optim, which='optim')


    widths = []
    the_keys_to_use = sorted(D.keys())
    print("K = ", the_keys_to_use)
    for width in the_keys_to_use:
        levels_to_plot = []
        metrics = defaultdict(lambda: [])
        widths.append(width)
        for d in D[width]:
            levels_to_plot.append(d['levels'][0])
            for key in d['eval_results_single']:
                if key != 'SolvabilityMetric':
                    V = np.mean(get_only_solvable_metrics(d['eval_results_all'][key], d['eval_results_all']['SolvabilityMetric']))
                else:
                    V = np.mean(d['eval_results_all'][key])
                metrics[key].append(V)
            for key in ['generation_time']:
                metrics[key].append(d[key])        
        for key in metrics:
            metrics[key] = np.array(metrics[key])
            all_values_mean[key].append(np.mean(metrics[key]))
            all_values_std[key].append(np.std(metrics[key]))
        
        dir = 'results/mario/206/line_graph/levels_neat'
        os.makedirs(dir, exist_ok=True)
        for i, l in enumerate(levels_to_plot):
            plt.figure(figsize=(40, 16))
            plt.imshow(l.show(False))
            plt.axis('off')
            plt.savefig(os.path.join(dir, f'{width}-{i}.png'), pad_inches=0.1, bbox_inches='tight')
            plt.close()
            
    metrics_to_plot = [
        'LeniencyMetric',
        'generation_time',
        'SolvabilityMetric',
        'CompressionDistanceMetric',
        'AStarDiversityMetric',
        'AStarDifficultyMetric',
        'AStarEditDistanceDiversityMetric'
    ]
    sns.set_theme()

    for key in metrics_to_plot:
        all_values_mean[key] = np.array(all_values_mean[key])
        all_values_std[key] = np.array(all_values_std[key])

        all_values_mean_direct_ga[key] = np.array(all_values_mean_direct_ga[key])
        all_values_std_direct_ga[key] = np.array(all_values_std_direct_ga[key])

        all_values_mean_direct_ga_optim[key] = np.array(all_values_mean_direct_ga_optim[key])
        all_values_std_direct_ga_optim[key] = np.array(all_values_std_direct_ga_optim[key])
        print(key)
        plt.figure()
        plt.plot(widths, all_values_mean[key], label='NoveltyNEAT (Ours)')
        plt.fill_between(widths, all_values_mean[key] - all_values_std[key], all_values_mean[key] + all_values_std[key], alpha=0.5)

        plt.plot(directga_widths, all_values_mean_direct_ga[key], label='DirectGA')
        plt.fill_between(directga_widths, all_values_mean_direct_ga[key] - all_values_std_direct_ga[key], all_values_mean_direct_ga[key] + all_values_std_direct_ga[key], alpha=0.5)

        plt.plot(directga_widths, all_values_mean_direct_ga_optim[key], label='DirectGA+')
        plt.fill_between(directga_widths, all_values_mean_direct_ga_optim[key] - all_values_std_direct_ga_optim[key], all_values_mean_direct_ga_optim[key] + all_values_std_direct_ga_optim[key], alpha=0.5)

        plt.xlabel("Level Width")
        pkey = pretty_key(key).replace("Metric", "")
        plt.ylabel(pkey)
        plt.title(f"Comparing {pkey} vs Level Size. Higher is better.")
        if 'time' in key.lower():
            plt.title(f"Comparing {pkey} vs Level Size. Lower is better.")
            # This is how long PCGRL took.
            plt.scatter([114, 171], [24 * 3600 * 3, 24 * 3600 * 3], marker='x', color='red', label='PCGRL')
            plt.yscale('log')
        if 'solv' in key.lower():
            plt.ylim(0, 1.1)
        plt.tight_layout()
        # plt.show()
        plt.legend()

        dir = 'results/mario/206/line_graph'
        os.makedirs(dir, exist_ok=True)
        plt.savefig(f'{dir}/{key}.png')


    df = pd.DataFrame(all_metrics).round(2)
    print(df)

if __name__ == '__main__':
    analyse_206_with_line_graph()
