"""
    This runs an experiment where we generate levels of different sizes and measure their metrics. 
    We do this for the Maze, using NoveltyNEAT. DirectGA is done in v107.
"""
import os
import pickle
from typing import List
from matplotlib import pyplot as plt
import neat
import numpy as np
import ray
from common.utils import get_date, save_compressed_pickle
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarEditDistanceDiversityMetric, AStarSolvabilityMetric
from metrics.average_pairwise_distance import AveragePairWiseDistanceMetric
from metrics.diversity.simple_diversity import EditDistanceMetric, HammingDistanceMetric
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.horn.linearity import LinearityMetric
from metrics.metric import Metric
from metrics.rl.tabular.rl_agent_metric import RLAgentMetric, compare_actions_edit_distance
from metrics.rl.tabular.rl_difficulty_metric import RLDifficultyMetric
from metrics.solvability import SolvabilityMetric
from novelty_neat.maze.neat_maze_level_generation import GenerateMazeLevelsUsingTiling
from timeit import default_timer as tmr

"""
    This file generates images and results showing that when training on small levels, we can still generate larger levels quite easily, that still
    maintain (or improve upon) the metric values.
"""


def exp_104_b():
    """This runs experiment 104 b, specifically testing levels with sizes 10 - 100 in increments of 10.

    Returns:
        [type]: [description]
    """
    import ray
    ray.init(num_cpus=10)
    
    print("HELLO, Running Experiment 104b")
    def evaluate_one_set_of_levels(metrics, levels, w, og_dic):
        val = {}
        val['w'] = w
        val['levels'] = levels
        val['eval_results_all'] = {}
        val['eval_results_single'] = {}
        
        # Add start and end if required
        for l in levels:
            if not hasattr(l, 'start'): l.start = (0, 0)
            height, width = l.map.shape
            if not hasattr(l, 'end'): l.end = (width - 1, height - 1)
        
        for myindex, m in enumerate(metrics):
            print(f"Running Metric {myindex+1} / {len(metrics)} -- {m.name()}")
            t_start = tmr()
            values = m.evaluate(levels)
            t_end = tmr()
            N = m.__class__.__name__
            N = m.name()
            val['eval_results_single'][N] = np.mean(values)
            val['eval_results_all'][N] = values
            
            print(f"{N:<30} OG = {og_dic['eval_results_single'].get(N, 'not there')}, now = {np.mean(values)}. Metric {N} with w={w} took {t_end - t_start}s")
        val['parent'] = metrics[0].parent
        print("--")
        return val
    names = [
        '../results/experiments/experiment_105_a/Maze/NEAT/2021-10-31_13-23-23/50/200/0/seed_0_name_experiment_105_a_2021-10-31_13-42-34.p',
        '../results/experiments/experiment_105_a/Maze/NEAT/2021-10-31_13-23-23/50/200/1/seed_1_name_experiment_105_a_2021-10-31_13-42-14.p',
        '../results/experiments/experiment_105_a/Maze/NEAT/2021-10-31_13-23-23/50/200/2/seed_2_name_experiment_105_a_2021-10-31_13-41-59.p',
        '../results/experiments/experiment_105_a/Maze/NEAT/2021-10-31_13-23-23/50/200/3/seed_3_name_experiment_105_a_2021-10-31_13-43-19.p',
        '../results/experiments/experiment_105_a/Maze/NEAT/2021-10-31_13-23-23/50/200/4/seed_4_name_experiment_105_a_2021-10-31_13-40-59.p',
    ]
    Ws = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000, 2000]
    config = './runs/proper_experiments/v100_maze/config/tiling_generate_12_1_balanced_pop100'
    all_dictionary = {
        'files': names,
        'config': config,
        'data': {
            w: [] for w in Ws
        },
        'original': []
    }
    I = 0

    def get_metrics(g: MazeGame):
        w = g.level.width
        parent = AStarDiversityAndDifficultyMetric(g, number_of_times_to_do_evaluation=5)

        solv = AStarSolvabilityMetric(g, parent)
        div = AStarDiversityMetric(g, parent)
        diff = AStarDifficultyMetric(g, parent)
        edit_distance_div = AStarEditDistanceDiversityMetric(g, parent)

        return [
                solv,
                diff,
                div,
                edit_distance_div,
                
                # Commented out 2022/03/31 to run faster for larger levels
                # LeniencyMetric(g),
                # LinearityMetric(g),
                # CompressionDistanceMetric(g),
                # AveragePairWiseDistanceMetric(g),
                # HammingDistanceMetric(g),
                # EditDistanceMetric(g),                
            ]

    @ray.remote
    def single_name_eval(I, name):
        print(f"{I} / 4")
        with open(name, 'rb') as f:
            dic = pickle.load(f)
        
        best = dic['train_results'][0]['final_agent']
        # print("ABC = ", dic['eval_results_single'])
        neat_conf = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config)
        net = neat.nn.FeedForwardNetwork.create(best, neat_conf)
        
        # Here, evaluate the original levels on all of the metrics.
        print("Evaluating original levels")
        g = MazeGame(MazeLevel())
        metrics = get_metrics(g)
        
        val_og = evaluate_one_set_of_levels(metrics, dic['levels'], 14, dic)
        # all_dictionary['original'].append(val_og)
        # END EVAL OG levels
        # all_dictionary['data'][w]
        all_others = {}
        for w in Ws:
            print(f"{w} -- {I}")
            h = w
            generator = GenerateMazeLevelsUsingTiling(game=MazeGame(MazeLevel(w, h)), number_of_random_variables=4, 
                    should_add_coords=False,
                    do_padding_randomly=False,
                    should_start_with_full_level=False, 
                    random_perturb_size=0.1565)

            levels = []
            time_start = tmr()
            for i in range(100):
                levels.append(generator(net))
            time_end = tmr()
            print(f"FOR LEVEL SIZE: {w} is gen time = {time_end - time_start}s")
            g = MazeGame(MazeLevel(w, h))
            metrics = get_metrics(g)
            val = evaluate_one_set_of_levels(metrics, levels, w, dic)
            val['generation_time'] = time_end - time_start
            
            all_others[w] = val
        return val_og, all_others

    futures = [single_name_eval.remote(I, name) for I, name in enumerate(names)]

    all_answers = ray.get(futures)
    for val_og, others in all_answers:
        all_dictionary['original'].append(val_og)
        for w, val in others.items():
            all_dictionary['data'][w].append(val)

        
    d = f'../results/experiments/104b/runs/{get_date()}'
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/data_large.p", 'wb+') as f:
        pickle.dump(all_dictionary, f)
        
    # Make dictionary a bit smaller as the large levels take up a lot of space
    for w, little_d in all_dictionary['data'].items():
        for p in little_d:
            del p['parent']
            if w > 100: 
                p['levels'] = p['levels'][:10]
        
    save_compressed_pickle(f"{d}/data", all_dictionary)

if __name__ == '__main__':
    exp_104_b()
