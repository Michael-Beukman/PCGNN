import os
import pickle
import sys
from matplotlib import pyplot as plt
import numpy as np
import wandb
from baselines.ga.multi_pop_ga.multi_population_mario import MarioGAPCG
from common.utils import get_date
from experiments.config import Config
from experiments.experiment import Experiment
from games.mario.mario_game import MarioGame
import neat
from games.mario.mario_level import MarioLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarEditDistanceDiversityMetric, AStarSolvabilityMetric
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from timeit import default_timer as tmr
import ray
from metrics.horn.linearity import LinearityMetric
from novelty_neat.general.neat_generate_general_level import GenerateGeneralLevelUsingTiling

def v206a_neat():
    np.random.seed(42)
    """this takes in the neat network and generates levels of different sizes
        without retraining.
    """
    """This runs experiment 206a, specifically testing levels with sizes 10 - 100 in increments of 10.

    """
    ray.init(num_cpus=5)
    
    print("HELLO, Running Experiment 206a")
    def evaluate_one_set_of_levels(metrics, levels, w, og_dic):
        val = {}
        val['w'] = w
        val['levels'] = levels
        val['eval_results_all'] = {}
        val['eval_results_single'] = {}
        for myindex, m in enumerate(metrics):
            print(f"Metric {myindex+1} / {len(metrics)}")
            values = m.evaluate(levels)
            N = m.__class__.__name__
            N = m.name()
            val['eval_results_single'][N] = np.mean(values)
            val['eval_results_all'][N] = values
            
            # For those not there...
            print(f"{N:<30} OG = {og_dic['eval_results_single'].get(N, 'not there')}, now = {np.mean(values)}")
        return val

    names = [
        '../results/experiments/experiment_204e/Mario/NeatNovelty/2021-10-25_20-04-31/100/150/True/1/0/6/2/-1/True/0/seed_0_name_experiment_204e_2021-10-25_23-53-36.p',
        '../results/experiments/experiment_204e/Mario/NeatNovelty/2021-10-25_20-04-31/100/150/True/1/0/6/2/-1/True/1/seed_1_name_experiment_204e_2021-10-25_23-33-22.p',
        '../results/experiments/experiment_204e/Mario/NeatNovelty/2021-10-25_20-04-31/100/150/True/1/0/6/2/-1/True/2/seed_2_name_experiment_204e_2021-10-25_23-32-37.p',
        '../results/experiments/experiment_204e/Mario/NeatNovelty/2021-10-25_20-04-31/100/150/True/1/0/6/2/-1/True/3/seed_3_name_experiment_204e_2021-10-25_23-50-26.p',
        '../results/experiments/experiment_204e/Mario/NeatNovelty/2021-10-25_20-04-31/100/150/True/1/0/6/2/-1/True/4/seed_4_name_experiment_204e_2021-10-25_23-59-43.p',
    ]


    Ws = [28, 56, 85, 114, 171, 228]
    config = './novelty_neat/configs/tiling_mario_56_7_1pred_size_one_hot_100_pop_clean'

    all_dictionary = {
        'files': names,
        'config': config,
        'data': {
            w: [] for w in Ws
        },
        'original': []
    }
    I = 0

    def get_metrics(g: MarioGame):
        w = g.level.width
        parent = AStarDiversityAndDifficultyMetric(g, number_of_times_to_do_evaluation=5)
        
        # potentially for longer levels we don't have enough time, but I think it is fine
        div = AStarDiversityMetric(g, parent)
        diff = AStarDifficultyMetric(g, parent)
        solv = AStarSolvabilityMetric(g, parent)
        return [
            solv,
            LeniencyMetric(g),
            CompressionDistanceMetric(g),
            CompressionDistanceMetric(g, use_combined_features=True),
            CompressionDistanceMetric(g, do_mario_flat=True),
            div,
            diff,
            AStarEditDistanceDiversityMetric(g, parent),
    ]

    @ray.remote
    def single_name_eval(I, name):
        print(f"{I} / 4")
        with open(name, 'rb') as f:
            dic = pickle.load(f)
        
        best = dic['train_results'][0]['final_agent']

        neat_conf = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config)
        net = neat.nn.FeedForwardNetwork.create(best, neat_conf)
        
        # Here, evaluate the original levels on all of the metrics.
        print("Evaluating original levels")
        g = MarioGame(MarioLevel())
        metrics = get_metrics(g)
        val_og = evaluate_one_set_of_levels(metrics, dic['levels'], 114, dic)
        
        all_others = {}
        for w in Ws:
            print(f"{w}")
            h = 14
            g = MarioGame(MarioLevel(w, h))
            
            generator = GenerateGeneralLevelUsingTiling(g, 1, 4, False, 0, 
                                                      predict_size=1, 
                                                      reversed_direction = 0, 
                                                      use_one_hot_encoding=True)

            levels = []
            time_start = tmr()
            for i in range(100):
                levels.append(generator(net))
            time_end = tmr()
            metrics = get_metrics(g)
            val = evaluate_one_set_of_levels(metrics, levels, w, dic)
            val['generation_time'] = (time_end - time_start) / 100
            print(f"FOR LEVEL SIZE: {w} is gen time = {val['generation_time']}s per level")
            
            all_others[w] = val
        return val_og, all_others

    futures = [single_name_eval.remote(I, name) for I, name in enumerate(names)]

    all_answers = ray.get(futures)
    for val_og, others in all_answers:
        all_dictionary['original'].append(val_og)
        for w, val in others.items():
            all_dictionary['data'][w].append(val)

    d = f'../results/experiments/206a/runs/{get_date()}'
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/data.p", 'wb+') as f:
        pickle.dump(all_dictionary, f)

def v206b_directga():
    """
    This tries to run
    """
    np.random.seed(42)
    date = get_date()
    ray.init(num_cpus=5)

    @ray.remote
    def single_func(seed, index, name, game, method, results_directory, args, date, get_pop, w):
        # A single function to run one seed and one param combination.
        proper_game = MarioGame(MarioLevel(w, 14))
        # get the metrics
        args['level_size_width'] = w
        print(f"proper_game_size = {proper_game.level.map.shape}")
        config = Config(
            name=name,
            game=game,
            method=method,
            seed=seed,
            results_directory=os.path.join(results_directory, str(seed)),
            method_parameters=args,
            date=date,
            project='NoveltyNeatPCG-Experiment206b'
        )
        print("Date = ", config.date, config.results_directory,
                config.hash(seed=False))
        parent = AStarDiversityAndDifficultyMetric(proper_game, number_of_times_to_do_evaluation=5)
        
        div = AStarDiversityMetric(proper_game, parent)
        diff = AStarDifficultyMetric(proper_game, parent)
        solv = AStarSolvabilityMetric(proper_game, parent)
        
        experiment = Experiment(config, get_pop, [
            CompressionDistanceMetric(proper_game),
            LinearityMetric(proper_game),
            LeniencyMetric(proper_game),
            solv,
            div,
            diff,
            AStarEditDistanceDiversityMetric(proper_game, parent),
        ], log_to_wandb=True)
        experiment.do_all()
        print(f"Generation time for size = {w} = {experiment.generation_time}")
        wandb.finish()

        return f"Completed index = {index} with seed = {seed}"
    
    def one_run(index, pop_size: int, num_gens: int, level_size: int):
        name = 'experiment_206b'
        game = 'Mario'
        method = 'DirectGA'
        generations = num_gens

        print(
            f"Doing now Size = {level_size}, Pop Size = {pop_size:>3}, Generations = {num_gens:>3}")
        results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{level_size}/{pop_size}/{generations}/'

        args = {
            'population_size': pop_size,
            'number_of_generations': generations,
            'level_gen': 'DirectGA',
            'level_size': level_size
        }

        def get_pop():
            level: MarioLevel = MarioLevel(level_size, 14)
            assert level.map.shape == (14, level_size)
            print("LEVEL Map size = ", level.map.shape)
            game = MarioGame(level)
            temp = MarioGAPCG(game, level, pop_size, generations, 
                        desired_entropy=0,
                        entropy_block_size=114,
                        ground_maximum_height=2,
                        
                        desired_sparseness_blocks=1,
                        blocks_block_size=10,


                        desired_sparseness_enemies=0.5,
                        enemies_block_size=20,

                        desired_sparseness_coins=0.5,
                        coin_maximum_height=2,
                        coins_block_size=10
                        )
            return temp

        futures = [single_func.remote(
            seed=i, index=index, name=name, game=game, method=method, results_directory=results_directory, args=args, date=date, get_pop=get_pop, w=level_size
        ) for i in range(5)]
        # We return the ray futures, and then only get them later, so that ray can run more than 5 at a time.
        return futures

    # The grid search thing

    Ws = [28, 56, 85, 114, 171, 228]
    counter = 0
    all_futures = []
    conf = {
            'pop_size': 20,
            'num_gens': 100,
        }
    for w in Ws:
        counter += 1
        all_futures += one_run(counter, level_size=w, **conf)
    print(f"At end, we have {counter} runs to do. Running Experiment 206b" )
    ray.get(all_futures)
    print("Done")


def v206c_optimised_directga():
    """
    This tries to run
    """
    np.random.seed(42)

    date = get_date()
    ray.init(num_cpus=5)

    @ray.remote
    def single_func(seed, index, name, game, method, results_directory, args, date, get_pop, w):
        # A single function to run one seed and one param combination.
        proper_game = MarioGame(MarioLevel(w, 14))
        # get the metrics
        args['level_size_width'] = w
        print(f"proper_game_size = {proper_game.level.map.shape}")
        config = Config(
            name=name,
            game=game,
            method=method,
            seed=seed,
            results_directory=os.path.join(results_directory, str(seed)),
            method_parameters=args,
            date=date,
            project='NoveltyNeatPCG-Experiment206c'
        )
        print("Date = ", config.date, config.results_directory,
                config.hash(seed=False))
        parent = AStarDiversityAndDifficultyMetric(proper_game, number_of_times_to_do_evaluation=5)
        div = AStarDiversityMetric(proper_game, parent)
        diff = AStarDifficultyMetric(proper_game, parent)
        solv = AStarSolvabilityMetric(proper_game, parent)
        
        experiment = Experiment(config, get_pop, [
            CompressionDistanceMetric(proper_game),
            LinearityMetric(proper_game),
            LeniencyMetric(proper_game),
            solv,
            div,
            diff,
            AStarEditDistanceDiversityMetric(proper_game, parent),
        ], log_to_wandb=True)
        experiment.do_all()
        print(f"Generation time for size = {w} = {experiment.generation_time}")
        wandb.finish()

        return f"Completed index = {index} with seed = {seed}"
    
    def one_run(index, pop_size: int, num_gens: int, level_size: int):
        name = 'experiment_206c'
        game = 'Mario'
        method = 'DirectGA'
        generations = num_gens

        print(
            f"Doing now Size = {level_size}, Pop Size = {pop_size:>3}, Generations = {num_gens:>3}")
        results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{level_size}/{pop_size}/{generations}/'
        
        ARGS = {'desired_entropy': 0.5, 'desired_sparseness_enemies': 0.5, 'desired_sparseness_coins': 0.5, 'desired_sparseness_blocks': 0.5, 'entropy_block_size': 20, 'enemies_block_size': 20, 'coins_block_size': 10, 'blocks_block_size': 40, 'ground_maximum_height': 2, 'coin_maximum_height': 2}
        args = {
            'population_size': pop_size,
            'number_of_generations': generations,
            'level_gen': 'DirectGA',
            'level_size': level_size,
            **ARGS
        }

        def get_pop():
            level: MarioLevel = MarioLevel(level_size, 14)
            assert level.map.shape == (14, level_size)
            print("LEVEL Map size = ", level.map.shape)
            game = MarioGame(level)
            temp = MarioGAPCG(game, level, pop_size, generations, 

                        **ARGS
                        )
            return temp

        futures = [single_func.remote(
            seed=i, index=index, name=name, game=game, method=method, results_directory=results_directory, args=args, date=date, get_pop=get_pop, w=level_size
        ) for i in range(5)]
        # We return the ray futures, and then only get them later, so that ray can run more than 5 at a time.
        return futures

    # The grid search thing

    Ws = [28, 56, 85, 114, 171, 228]
    counter = 0
    all_futures = []
    conf = {
            'pop_size': 10,
            'num_gens': 50,
        }
    for w in Ws:
        counter += 1
        all_futures += one_run(counter, level_size=w, **conf)
    print(f"At end, we have {counter} runs to do. Running Experiment 206c" )
    ray.get(all_futures)
    print("Done")


if __name__ == '__main__':
    I = int(sys.argv[-1])
    if I == 0:
        print("DOING NEAT")
        v206a_neat()
    elif I == 1:
        print("DOING DIRECT GA")
        v206b_directga()
    elif I == 2:
        print("DOING OPTIMISED DIRECT GA")
        v206c_optimised_directga()
