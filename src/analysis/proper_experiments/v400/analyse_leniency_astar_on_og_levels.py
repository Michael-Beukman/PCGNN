import os
from matplotlib import pyplot as plt
import numpy as np
from games.mario.java_runner import java_astar_number_of_things_in_open_set, run_java_task
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
from metrics.horn.leniency import LeniencyMetric
from scipy.stats import pearsonr

def astar():
    xs = []
    diffs = []
    solvs = []
    good_indices = [ 1, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15,]
    DIR = 'results/v400/metrics/og_levels'
    os.makedirs(DIR, exist_ok=True)
    J = 0
    for i in range(1, 16):
        if i not in good_indices: continue

        # break
        level = f'external/Mario-AI-Framework/levels/original/lvl-{i}.txt'
        for P in range(1):
            is_solv, a, ab, metric = java_astar_number_of_things_in_open_set(None, filename=level)
            if is_solv: break
        
        if is_solv or 1:
            xs.append(J)
            diffs.append(metric)
            solvs.append(is_solv + 1)
            print(f'lvl {i}, AStar Difficulty (unnormalised) = {metric}. Solv = {solvs[-1]}, took {P} steps')
            J += 1
    T = np.array(solvs) * 1.0 * np.max(diffs)
    print(diffs)
    print(solvs)

    plt.plot(xs, T, label='Solvability')
    plt.scatter(xs, diffs, label='A* Diff')
    r, p = pearsonr(xs, diffs)
    plt.title(f"Original Level index vs difficulty. R = {r}, P = {p}")
    plt.xlabel("Original Level index")
    # plt.yscale('log')
    plt.ylabel("A* Difficulty")
    plt.legend()
    plt.savefig(os.path.join(DIR, 'astar.png'))
    # plt.show()

def leniency():
    DIR = 'results/v400/metrics/og_levels'
    os.makedirs(DIR, exist_ok=True)
    with open ('analysis/horn_levels/good_og_leniency.csv', 'r') as f:
        lines = f.readlines()
    Xs = []
    Lens = []
    for i, l in enumerate(lines):
        name, len = l.strip().split(",")
        len = float(len)
        Lens.append(len)
        Xs.append(i)
    plt.scatter(Xs, Lens)
    r, p = pearsonr(Xs, Lens)
    plt.title(f"Original Level index vs difficulty. R = {r}, P = {p}")
    plt.xlabel("Original Level index")
    # plt.yscale('log')
    plt.ylabel("Leniency")
    plt.legend()
    plt.savefig(os.path.join(DIR, 'leniency.png'))
    # plt.show()




if __name__ == '__main__':
    # main()
    astar()
    plt.close()
    leniency()