"""
This is an experiment where we run our and other metrics across many different levels and compare the following:
- Our diversity vs compression distance
- Our Difficulty vs Leniency
"""

import glob
import os
import pickle
from typing import List
from common.utils import get_date

from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarEditDistanceDiversityMetric
from metrics.average_pairwise_distance import AveragePairWiseDistanceMetric
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.metric import Metric
from metrics.rl.tabular.rl_agent_metric import RLAgentMetric
from metrics.rl.tabular.rl_difficulty_metric import RLDifficultyMetric
from metrics.solvability import SolvabilityMetric
import ray
from analysis.proper_experiments.v100.analyse_106 import main as analyse_things


def main():
    ray.init()
    pickles = [
        '../results/v2_test_ga/Maze/DirectGANoveltySolvability/2021-07-03_10-03-41/50/100/3/seed_3_name_v2_test_ga_2021-07-03_10-16-06.p',
        '../results/v2_test_ga/Maze/DirectGANoveltySolvability/2021-07-03_10-03-41/50/100/2/seed_2_name_v2_test_ga_2021-07-03_10-15-55.p',
        '../results/v2_test_ga/Maze/DirectGANoveltySolvability/2021-07-03_10-03-41/50/100/1/seed_1_name_v2_test_ga_2021-07-03_10-15-54.p',
        '../results/v2_test_ga/Maze/DirectGANoveltySolvability/2021-07-03_10-03-41/50/100/0/seed_0_name_v2_test_ga_2021-07-03_10-16-01.p',
    ] + [
        '../results/experiments/experiment_105_a/Maze/NEAT/2021-08-19_07-16-59/100/139/0/seed_0_name_experiment_105_a_2021-08-19_09-54-05.p',
        '../results/experiments/experiment_105_a/Maze/NEAT/2021-08-19_07-16-59/100/139/1/seed_1_name_experiment_105_a_2021-08-19_09-59-59.p',
        '../results/experiments/experiment_105_a/Maze/NEAT/2021-08-19_07-16-59/100/139/2/seed_2_name_experiment_105_a_2021-08-19_09-55-52.p',
        '../results/experiments/experiment_105_a/Maze/NEAT/2021-08-19_07-16-59/100/139/3/seed_3_name_experiment_105_a_2021-08-19_09-57-12.p',
        '../results/experiments/experiment_105_a/Maze/NEAT/2021-08-19_07-16-59/100/139/4/seed_4_name_experiment_105_a_2021-08-19_09-58-24.p',
    ] + [
        '../results/v2_test_ga/Maze/DirectGAEntropySolvability/2021-07-31_08-56-26/10/100/3/seed_3_name_v2_test_ga_2021-07-31_09-14-05.p',
        '../results/v2_test_ga/Maze/DirectGAEntropySolvability/2021-07-31_08-56-26/10/100/2/seed_2_name_v2_test_ga_2021-07-31_09-14-10.p',
        '../results/v2_test_ga/Maze/DirectGAEntropySolvability/2021-07-31_08-56-26/10/100/1/seed_1_name_v2_test_ga_2021-07-31_09-14-04.p',
        '../results/v2_test_ga/Maze/DirectGAEntropySolvability/2021-07-31_08-56-26/10/100/0/seed_0_name_v2_test_ga_2021-07-31_09-14-31.p',
    ] + [
        'results/maze/pcgrl/turtle/train_proper_turtle_0917/run_seed_1.p',
        'results/maze/pcgrl/turtle/train_proper_turtle_0917/run_seed_2.p',
        'results/maze/pcgrl/turtle/train_proper_turtle_0917/run_seed_3.p',
        'results/maze/pcgrl/turtle/train_proper_turtle_0917/run_seed_4.p',
        'results/maze/pcgrl/turtle/train_proper_turtle_0917/run_seed_5.p',
    ] + [
        'results/maze/pcgrl/wide/train_proper_wide_0915/run_seed_1.p',
        'results/maze/pcgrl/wide/train_proper_wide_0915/run_seed_2.p',
        'results/maze/pcgrl/wide/train_proper_wide_0915/run_seed_3.p',
        'results/maze/pcgrl/wide/train_proper_wide_0915/run_seed_4.p',
        'results/maze/pcgrl/wide/train_proper_wide_0915/run_seed_5.p',
    ] + [
        '../results/v4_test_neat_sweep_more_test_things/Maze/NeatNoveltySolvabilityRandomPadding/2021-08-13_08-11-07/100/0/seed_0_name_v4_test_neat_sweep_more_test_things_2021-08-13_09-36-53.p',
        '../results/v4_test_neat_sweep_more_test_things/Maze/NeatNoveltySolvabilityRandomPadding/2021-08-13_08-11-07/100/1/seed_1_name_v4_test_neat_sweep_more_test_things_2021-08-13_11-09-00.p',
        '../results/v4_test_neat_sweep_more_test_things/Maze/NeatNoveltySolvabilityRandomPadding/2021-08-13_08-11-07/100/2/seed_2_name_v4_test_neat_sweep_more_test_things_2021-08-13_12-34-17.p',
        '../results/v4_test_neat_sweep_more_test_things/Maze/NeatNoveltySolvabilityRandomPadding/2021-08-13_08-11-07/100/3/seed_3_name_v4_test_neat_sweep_more_test_things_2021-08-13_14-03-46.p',
    ] + [
        '../results/experiments/experiment_102_d_visual_diversity/Maze/DirectGA/2021-09-29_07-25-28/100/100/1/True/True/0/seed_0_name_experiment_102_d_visual_diversity_2021-09-29_09-49-47.p',
        '../results/experiments/experiment_102_d_visual_diversity/Maze/DirectGA/2021-09-29_07-25-28/100/100/1/True/True/1/seed_1_name_experiment_102_d_visual_diversity_2021-09-29_10-40-47.p',
        '../results/experiments/experiment_102_d_visual_diversity/Maze/DirectGA/2021-09-29_07-25-28/100/100/1/True/True/2/seed_2_name_experiment_102_d_visual_diversity_2021-09-29_09-01-09.p',
        '../results/experiments/experiment_102_d_visual_diversity/Maze/DirectGA/2021-09-29_07-25-28/100/100/1/True/True/3/seed_3_name_experiment_102_d_visual_diversity_2021-09-29_11-27-20.p',
        '../results/experiments/experiment_102_d_visual_diversity/Maze/DirectGA/2021-09-29_07-25-28/100/100/1/True/True/4/seed_4_name_experiment_102_d_visual_diversity_2021-09-29_11-09-06.p',
    ] + [
        '../results/experiments/experiment_102_f_visual_diversity_rerun_batch/Maze/DirectGA/2021-10-03_08-34-50/50/100/0/True/True/0/seed_0_name_experiment_102_f_visual_diversity_rerun_batch_2021-10-03_09-12-00.p',
        '../results/experiments/experiment_102_f_visual_diversity_rerun_batch/Maze/DirectGA/2021-10-03_08-34-50/50/100/0/True/True/1/seed_1_name_experiment_102_f_visual_diversity_rerun_batch_2021-10-03_09-21-28.p',
        '../results/experiments/experiment_102_f_visual_diversity_rerun_batch/Maze/DirectGA/2021-10-03_08-34-50/50/100/0/True/True/2/seed_2_name_experiment_102_f_visual_diversity_rerun_batch_2021-10-03_08-53-46.p',
        '../results/experiments/experiment_102_f_visual_diversity_rerun_batch/Maze/DirectGA/2021-10-03_08-34-50/50/100/0/True/True/3/seed_3_name_experiment_102_f_visual_diversity_rerun_batch_2021-10-03_09-11-53.p',
        '../results/experiments/experiment_102_f_visual_diversity_rerun_batch/Maze/DirectGA/2021-10-03_08-34-50/50/100/0/True/True/4/seed_4_name_experiment_102_f_visual_diversity_rerun_batch_2021-10-03_09-32-38.p',
    ] + [
            '../results/experiments/experiment_102_aaa_rerun_only_best/Maze/DirectGA/2021-09-16_07-51-09/100/100/1/True/False/0/seed_0_name_experiment_102_aaa_rerun_only_best_2021-09-16_08-18-25.p',
            '../results/experiments/experiment_102_aaa_rerun_only_best/Maze/DirectGA/2021-09-16_07-51-09/100/100/1/True/False/1/seed_1_name_experiment_102_aaa_rerun_only_best_2021-09-16_08-18-19.p',
            '../results/experiments/experiment_102_aaa_rerun_only_best/Maze/DirectGA/2021-09-16_07-51-09/100/100/1/True/False/2/seed_2_name_experiment_102_aaa_rerun_only_best_2021-09-16_08-19-23.p',
            '../results/experiments/experiment_102_aaa_rerun_only_best/Maze/DirectGA/2021-09-16_07-51-09/100/100/1/True/False/3/seed_3_name_experiment_102_aaa_rerun_only_best_2021-09-16_08-18-30.p',
            '../results/experiments/experiment_102_aaa_rerun_only_best/Maze/DirectGA/2021-09-16_07-51-09/100/100/1/True/False/4/seed_4_name_experiment_102_aaa_rerun_only_best_2021-09-16_08-18-01.p',
            '../results/experiments/experiment_102_aaa_rerun_only_best/Maze/DirectGA/2021-09-16_07-51-09/100/100/1/True/True/0/seed_0_name_experiment_102_aaa_rerun_only_best_2021-09-16_13-13-52.p',
            '../results/experiments/experiment_102_aaa_rerun_only_best/Maze/DirectGA/2021-09-16_07-51-09/100/100/1/True/True/1/seed_1_name_experiment_102_aaa_rerun_only_best_2021-09-16_13-18-39.p',
            '../results/experiments/experiment_102_aaa_rerun_only_best/Maze/DirectGA/2021-09-16_07-51-09/100/100/1/True/True/2/seed_2_name_experiment_102_aaa_rerun_only_best_2021-09-16_13-04-45.p',
            '../results/experiments/experiment_102_aaa_rerun_only_best/Maze/DirectGA/2021-09-16_07-51-09/100/100/1/True/True/3/seed_3_name_experiment_102_aaa_rerun_only_best_2021-09-16_13-12-30.p',
            '../results/experiments/experiment_102_aaa_rerun_only_best/Maze/DirectGA/2021-09-16_07-51-09/100/100/1/True/True/4/seed_4_name_experiment_102_aaa_rerun_only_best_2021-09-16_13-17-46.p',
    ]
    print("LEN = ", len(pickles))
    game = MazeGame(MazeLevel())

    def get_metrics(g):
        parent = AStarDiversityAndDifficultyMetric(g, number_of_times_to_do_evaluation=5)

        div = AStarDiversityMetric(g, parent)
        diff = AStarDifficultyMetric(g, parent)
        return [
            SolvabilityMetric(g),
            LeniencyMetric(g),
            CompressionDistanceMetric(g),
            AveragePairWiseDistanceMetric(g),
            div,
            diff,
            AStarEditDistanceDiversityMetric(g, parent)
        ], parent


    @ray.remote
    def single_func(index: int, pickle_name: str):
        metrics, parent = get_metrics(game)
        mydic = {
            'name': pickle_name
        }
        with open(pickle_name, 'rb') as f:
            ans = pickle.load(f)
            levels = ans['levels']
        for k, metric in enumerate(metrics):
            print(
                f"Index {index} starting with metric {metric.name()} {k+1} / {len(metrics)}")
            mydic[metric.name()] = metric.evaluate(levels)
        mydic['test_parent'] = parent
        return mydic

    futures = [single_func.remote(i, pickles[i]) for i in range(len(pickles))]
    all_values = (ray.get(futures))

    d = f'../results/experiments/106/runs/{get_date()}'
    os.makedirs(d, exist_ok=True)
    FILENAME = f"{d}/data.p"
    with open(FILENAME, 'wb+') as f:
        pickle.dump(all_values, f)
    return FILENAME


if __name__ == '__main__':
    FILE = main()

    import matplotlib.pyplot as plt
    analyse_things(
        [
            'LeniencyMetric',
            'AStarDifficultyMetric'
        ],
        file_to_use=FILE
    )
    plt.close()
    analyse_things([
        'CompressionDistanceMetric',
        'EditDistanceMetric', 'HammingDistanceMetric',
        'AStarDiversityMetric'
    ], 'diversity', file_to_use=FILE)
