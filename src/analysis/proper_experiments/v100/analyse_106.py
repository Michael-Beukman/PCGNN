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

    path = file_to_use

    points = defaultdict(lambda: [])
    all_levels = []
    all_scores = []
    test_rld = []
    test_astar = []
    test_cd = []
    all_solvs = []
    with open(path, 'rb') as f:
        ls = pickle.load(f)
        I = -1
        for l in ls:
            I += 1
            # if I < 11: continue;
            print(l.keys())
            print("HELLO: ")
            with open (l['name'], 'rb') as f_:
                levels = pickle.load(f_)['levels']
            
            cd_values = np.zeros((100, 100))
            rld_values = np.zeros((100, 100))
            
            for i, c in enumerate(l['CompressionDistanceMetric']):
                cd_values[i // 100][i % 100] = c
                cd_values[i % 100][i // 100] = c
            
            for i, rld in enumerate(l['RLAgentMetric']):
                rld_values[i % 100][i // 100] = rld
                rld_values[i // 100][i % 100] = rld
            all_levels.extend(levels)
            for i in range(100):
                all_solvs.extend(l['SolvabilityMetric'])
            test_rld.append(rld_values)
            test_cd.append(cd_values)
    test_rld = np.array(test_rld) 
    test_cd = np.array(test_cd)
    all_solvs = np.array(all_solvs).reshape(test_cd.shape)

    highest_rl_idx = np.argmax(test_rld * all_solvs)
    lowest_rl_idx = np.argmin(test_rld * all_solvs)
    P = test_rld * all_solvs
    lowest_rl_idx = np.where(P == np.min(P[np.nonzero(P)]))
    lowest_rl_idx_rs = [lowest_rl_idx[0][0], lowest_rl_idx[1][0], lowest_rl_idx[2][0]]
    
    highest_rl_idx_rs = np.unravel_index(highest_rl_idx, test_rld.shape)

    offset = highest_rl_idx_rs[0] * 100
    levels = [
        all_levels[offset + highest_rl_idx_rs[1]],
        all_levels[offset + highest_rl_idx_rs[2]],
    ]
        
    offset = lowest_rl_idx_rs[0] * 100
    levels += [
        all_levels[offset + lowest_rl_idx_rs[1]],
        all_levels[offset + lowest_rl_idx_rs[2]],
    ]
    
    fig, axs = plt.subplots(len(levels) + 1, len(levels) + 1, figsize=(20,20))
    game = MazeGame(MazeLevel())

    metrics = [
        RLAgentMetric(game),
        CompressionDistanceMetric(game)
    ]
    traj, actions = metrics[0]._get_action_trajectories(levels, True)
    for a in axs.ravel():
        a.set_xticks([]),a.set_yticks([])
    for i, l in enumerate(levels):
        for a in [axs[0, i + 1], axs[i+1, 0]]:
            a.imshow(1 - l.map, vmin=0, vmax=1, cmap='gray')
            x, y = zip(*traj[i])
            a.plot(list(x), list(y))
    for i in range(len(traj)):
        for j in range(i, len(traj)):
            Ts = [traj[i], traj[j]]
            As = [actions[i], actions[j]]
            Ls = [levels[i], levels[j]]
            labels = []
            ys = []
            for I, m in enumerate(metrics):
                # Use actions when necessary.
                if I == 0:
                    V = m.evaluate(Ls, Ts)
                else:
                    V = m.evaluate(Ls)
                assert len(V) == 1
                V = V[0]
                ys.append(V)
                labels.append(m.name())
            axs[i + 1, j + 1].imshow(np.ones((14, 14)), cmap='gray', vmin=0, vmax=1)
            for T in Ts:
                x, y = zip(*T)
                axs[i + 1, j + 1].plot(list(x), list(y))
            S = ""
            for value, label in zip(ys, labels):
                S += f"{label.split('(')[0]}: {np.round(value, 2)}\n"
            axs[i + 1, j + 1].set_title(S[:-1])
            if i != j:
                fig.delaxes(axs[j + 1, i + 1])
    fig.delaxes(axs[0, 0])
    plt.tight_layout()
    dir = './results/maze/106'
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, 'diversity_comparison_levels' + ".png"))

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