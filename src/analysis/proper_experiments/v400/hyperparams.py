
import glob
import os
import pickle
import pprint
from typing import Any, Dict

import pandas as pd

def _get_neat_105a_hps():
    return {
            'Context Size': 1,
            'Predict Size': 1,
            'Number of Random Variables:': 4,
            'Padding': -1,
            'Random Perturb Size': 0.1565,


            # method,
            'Number of Generations': 200,
            'Population Size': 50,


            'Number of Levels': 24,
            'Number of Neighbours for Novelty Calculation': 15,
            '$\lambda$': 0,
            'Novelty Distance Function': 'Visual Diversity, only on reachable tiles',

            'Fitness': r'\begin{flushleft}Novelty() * 0.399\\ Solvability(Ignore First Tile) * 0.202\\ Novelty-Intra-Generator(neighbours=10) * 0.399\end{flushleft}',
            }
def _get_directga_plus_maze_102_aaa():
    # This one:
    # ../results/experiments/experiment_102_aaa_rerun_only_best/Maze/DirectGA/2021-10-25_20-05-42/100/100/1/True/False/:
    return {
        'Population Size': 100,
        'Number of Generations': 100,
        'Fitness': r'\begin{flushleft} Entropy(Desired Entropy=1) * 0.5\\ PartialSolvability() * 0.5\end{flushleft}',
    }


def _get_directga_novelty_maze_102_f():
    # This one:
    # ../results/experiments/experiment_102_f_visual_diversity_rerun_batch/Maze/DirectGA/2021-10-25_20-03-11/50/100/0/True/True/
    return {
        'Population Size': 50,
        'Number of Generations': 100,
        'Fitness': r'\begin{flushleft}Entropy(Desired Entropy=0) * 0.33 \\ PartialSolvability() * 0.33\\ Novelty(Visual Diversity, $\lambda=1$, number of neighbours=15) * 0.33 \end{flushleft}',
    }


def _get_neat_204e_hps():
    return {
            'Context Size': 1,
            'Predict Size': 1,
            'One Hot Inputs': 'Yes',
            'Number of Random Variables:': 4,
            'Padding': -1,
            'Random Perturb Size': 0,


            # method,
            'Number of Generations': 150,
            'Population Size': 100,


            'Number of Levels': 6,
            'Number of Neighbours for Novelty Calculation': 15,
            '$\lambda$': 0,
            'Novelty Distance Function': 'Visual Diversity',

            'Fitness': r'\begin{flushleft}Novelty() * 0.25 \\ Solvability() * 0.5 \\ Novelty-Intra-Generator(neighbours=2) * 0.25 \end{flushleft}',
            }

def _get_directga_optimised_201a():
    # ../results/experiments/experiment_201_a/Mario/DirectGA/2021-10-29_05-20-30/10/50/0.5/0.5/0.5/0.5/20/20/10/40/2/2/
    params = {'Population Size': '10', 'Number of Generations': '50', 'Desired Entropy': '0.5', 'desired_sparseness_enemies': '0.5', 'desired_sparseness_coins': '0.5', 'desired_sparseness_blocks': '0.5', 'entropy_block_size': '20', 'enemies_block_size': '20', 'coins_block_size': '10', 'blocks_block_size': '40', 'ground_maximum_height': '2'}
    new_params = {' '.join(k.split('_')).title().replace("Of", 'of'): v for k, v in params.items()}
    return new_params

def _get_directga_default_201b():
    # ../results/experiments/experiment_201_b/Mario/DirectGA/2021-10-25_20-03-10/20/100/0.0/0.0/1.0/0.5/114/20/10/10/2/2
    params = {'population_size': '20', 'number of generations': '100', 'desired_entropy': '0.0', 'desired_sparseness_enemies': '0.0', 'desired_sparseness_coins': '1.0', 'desired_sparseness_blocks': '0.5', 'entropy_block_size': '114', 'enemies_block_size': '20', 'coins_block_size': '10', 'blocks_block_size': '10', 'ground_maximum_height': '2'}
    new_params = {' '.join(k.split('_')).title().replace("Of", 'of'): v for k, v in params.items()}
    return new_params

def _get_directga_nov_201d():
    # '../results/experiments/experiment_201_d/Mario/DirectGA/2021-10-25_20-04-38/100/100/0.0/0.0/1.0/0.5/114/20/10/10/2/2/use_novelty'
    params = {'population_size': '100', 'number of generations': '100', 'desired_entropy': '0.0', 'desired_sparseness_enemies': '0.0', 'desired_sparseness_coins': '1.0', 'desired_sparseness_blocks': '0.5', 'entropy_block_size': '114', 'enemies_block_size': '20', 'coins_block_size': '10', 'ground_maximum_height': '2', 'blocks_block_size': '10', 'Novelty Distance Function': 'Visual Diversity', 'Novelty Neighbours': '6', 'Novelty $\lambda$':1}
    new_params = {' '.join(k.split('_')).title().replace("Of", 'of'): v for k, v in params.items()}
    return new_params

def list_hyperparameters():
    # This is going to be somewhat fun-ish.
    dic_of_all_hyperparams_maze = {
        'NoveltyNEAT': _get_neat_105a_hps(),
        'DirectGA+': _get_directga_plus_maze_102_aaa(),
        'DirectGA_Novelty': _get_directga_novelty_maze_102_f(),
        'PCGRL_Wide': {
            'Training Steps': 100_000_000,
            'Reward': 'Positive reward for having only one connected region of open tiles, and positive reward for having a path between the start and goal of at least 20, at most 80. Weighted these in a ratio of 5:2. Negative reward (-100) for a action that removed an existing path.'
        },
    }   

    dic_of_all_hyperparams_mario = {
        'NoveltyNEAT': _get_neat_204e_hps(),
        'DirectGA': _get_directga_default_201b(),
        'DirectGA+': _get_directga_optimised_201a(),
        'DirectGA_Novelty': _get_directga_nov_201d(),
        'PCGRL_Wide': {
            'Training Steps': 100_000_000,
            'Reward': 'Positive reward for having only one connected region of open tiles, and positive reward for having a path between the start and goal of at least 20, at most 80. Weighted these in a ratio of 5:2. Negative reward (-100) for a action that removed an existing path.'
        },
        
        'PCGRL_Turtle': {
            'Training Steps': 100_000_000,
            'Reward': 'Positive reward for having only one connected region of open tiles, and positive reward for having a path between the start and goal of at least 20, at most 80. Weighted these in a ratio of 5:2. Negative reward (-100) for a action that removed an existing path.'
        },
    }   

    for mode in ['all']:
        pd.set_option('display.max_colwidth', None)
        # mode = 'maze'
        dic_of_hps = dic_of_all_hyperparams_maze if mode in ['maze', 'all'] else dic_of_all_hyperparams_mario
        dir = f'results/v400/hyperparams/{mode}'
        os.makedirs(dir, exist_ok=True)
        for name, hps in dic_of_hps.items():
            if mode == 'all':
                df = pd.DataFrame({k: {'Maze': v, 'Mario': dic_of_all_hyperparams_mario[name].get(k, 'Null')} for k, v in hps.items()}).T
            else:
                df = pd.DataFrame({k: {'Value': v} for k, v in hps.items()}).T
            if mode == 'all': 
                print(f"Writing {name} with ")
            df.to_latex(os.path.join(dir, name + "_hps.tex"), escape=False, column_format='p{0.3\linewidth}' + '|p{0.35\linewidth}|p{0.35\linewidth}')
    
    dir = f'results/v400/hyperparams/maze'
    names = ['DirectGA+', 'DirectGA_Novelty']
    D = {key: {n.replace("_", ' '): 
        
        dic_of_all_hyperparams_maze[n][key] for n in names} 
        for key, v in dic_of_all_hyperparams_maze[names[0]].items()}
    df = pd.DataFrame(D).T
    df.to_latex(os.path.join(dir, "DirectGA_Combined" + "_hps.tex"), escape=False, column_format='p{0.3\linewidth}' + 'p{0.35\linewidth}p{0.35\linewidth}')


    dir = f'results/v400/hyperparams/mario'
    names = ['DirectGA+', 'DirectGA_Novelty', 'DirectGA']
    d = {key: {n.replace("_", ' '): 
            dic_of_all_hyperparams_mario[n][key] for n in names} 
            for key, v in dic_of_all_hyperparams_mario[names[0]].items()}
    df = pd.DataFrame(d).T
    df.to_latex(os.path.join(dir, "DirectGA_Combined" + "_hps.tex"), escape=False, column_format='p{0.4\linewidth}' + 'p{0.2\linewidth}p{0.2\linewidth}p{0.2\linewidth}')

if __name__ == '__main__':
    list_hyperparameters()
