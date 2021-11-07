from math import ceil
import os
import sys
import ray
import wandb
from baselines.ga.multi_pop_ga.multi_population_mario import MarioGAPCG
from common.utils import get_date
from experiments.config import Config
from experiments.experiment import Experiment
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarEditDistanceDiversityMetric, AStarSolvabilityMetric
from metrics.average_pairwise_distance import AveragePairWiseDistanceMetric, AveragePairWiseDistanceMetricOnlyPlayable
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.horn.linearity import LinearityMetric
from metrics.rl.tabular.rl_agent_metric import RLAgentMetric
from metrics.rl.tabular.rl_difficulty_metric import RLDifficultyMetric
from metrics.solvability import SolvabilityMetric
os.environ['WANDB_SILENT'] = 'True'


def experiment_201_a():
    """This is experiment 201 (a). 
         - This runs the direct GA on Mario with its normal entropy fitness function, over a range of parameters,
         specifically those specified in the constructor of this function.

         Deliberate choice to not use novelty,
            because we use the separate populations.
            Could do it later though.
    """
    # One date
    date = get_date()
    ray.init(num_cpus=14)

    @ray.remote
    def single_func(seed, index, name, game, method, results_directory, args, date, get_pop):
        # A single function to run one seed and one param combination.
        config = Config(
            name=name,
            game=game,
            method=method,
            seed=seed,
            results_directory=os.path.join(results_directory, str(seed)),
            method_parameters=args,
            date=date,
            project='NoveltyNeatPCG-Experiment201-a'
        )
        print("Date = ", config.date, config.results_directory,
              config.hash(seed=False))
        proper_game = MarioGame(MarioLevel())
        experiment = Experiment(config, get_pop, [
            CompressionDistanceMetric(proper_game),
            LinearityMetric(proper_game),
            LeniencyMetric(proper_game),
            SolvabilityMetric(proper_game),
            AveragePairWiseDistanceMetric(proper_game),
            AveragePairWiseDistanceMetricOnlyPlayable(proper_game),
        ], log_to_wandb=True)
        experiment.do_all()
        wandb.finish()

        return f"Completed index = {index} with seed = {seed}"

    def one_run(index,  **kwargs):
        name = 'experiment_201_a'
        game = 'Mario'
        method = 'DirectGA'
        generations = kwargs['num_gens']
        keys = ['pop_size', 'num_gens', 'desired_entropy', 'desired_sparseness_enemies', 'desired_sparseness_coins', 'desired_sparseness_blocks',
                'entropy_block_size', 'enemies_block_size', 'coins_block_size', 'blocks_block_size', 'ground_maximum_height', 'coin_maximum_height', ]
        print(
            f"Doing now, with params = {' | '.join([f'{k}:{v}' for k, v in kwargs.items()])}")
        results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{"/".join([str(kwargs[key]) for key in keys])}'
        print(f"Results directory = {results_directory}")

        all_dic_args = dict(pop_size=kwargs['pop_size'],
                            number_of_generations=kwargs['num_gens'],
                            desired_entropy=kwargs['desired_entropy'],
                            desired_sparseness_enemies=kwargs['desired_sparseness_enemies'],
                            desired_sparseness_coins=kwargs['desired_sparseness_coins'],
                            desired_sparseness_blocks=kwargs['desired_sparseness_blocks'],
                            entropy_block_size=kwargs['entropy_block_size'],
                            enemies_block_size=kwargs['enemies_block_size'],
                            coins_block_size=kwargs['coins_block_size'],
                            blocks_block_size=kwargs['blocks_block_size'],
                            ground_maximum_height=kwargs['ground_maximum_height'],
                            coin_maximum_height=kwargs['coin_maximum_height'],)
        args = {
            'population_size': kwargs['pop_size'],
            'number_of_generations': generations,
            'desired_entropy': kwargs['desired_entropy'],
            'level_gen': 'DirectGA',
            **all_dic_args
        }

        def get_pop():
            level = MarioLevel()
            game = MarioGame(level)
            return MarioGAPCG(game, level,
                              **all_dic_args
                              )

        futures = [single_func.remote(
            seed=i, index=index, name=name, game=game, method=method, results_directory=results_directory, args=args, date=date, get_pop=get_pop
        ) for i in range(5)]
        # We return the ray futures, and then only get them later, so that ray can run more than 5 at a time.
        return futures

    # The grid search thing

    counter = 0
    all_futures = []
    counter += 1
    params = {'Population Size': '10', 'Number of Generations': '50', 'Desired Entropy': '0.5', 'desired_sparseness_enemies': '0.5', 'desired_sparseness_coins': '0.5', 'desired_sparseness_blocks': '0.5', 'entropy_block_size': '20', 'enemies_block_size': '20', 'coins_block_size': '10', 'blocks_block_size': '40', 'ground_maximum_height': '2'}
    all_futures += one_run(counter,
                            pop_size=10,
                            num_gens=50,
                            desired_entropy=0.5,
                            desired_sparseness_enemies=0.5,
                            desired_sparseness_coins=0.5,
                            desired_sparseness_blocks=0.5,
                            entropy_block_size=20,
                            enemies_block_size=20,
                            coins_block_size=10,
                            blocks_block_size=40,
                            ground_maximum_height=2,
                            coin_maximum_height=2,
                            )
    print(f"At end, we have {counter} runs to do. Running experiment_201a")
    ray.get(all_futures)
    print("Done")

def experiment_201_b():
    # Does the thing as specified by the ferreira paper.
    # i.e. using the same hyperparams.
    date = get_date()
    ray.init(num_cpus=14)

    @ray.remote
    def single_func(seed, index, name, game, method, results_directory, args, date, get_pop):
        # A single function to run one seed and one param combination.
        config = Config(
            name=name,
            game=game,
            method=method,
            seed=seed,
            results_directory=os.path.join(results_directory, str(seed)),
            method_parameters=args,
            date=date,
            project='NoveltyNeatPCG-Experiment201-b'
        )
        print("Date = ", config.date, config.results_directory,
              config.hash(seed=False))
        proper_game = MarioGame(MarioLevel())
        experiment = Experiment(config, get_pop, [
            CompressionDistanceMetric(proper_game),
            LinearityMetric(proper_game),
            LeniencyMetric(proper_game),
            SolvabilityMetric(proper_game),
            AveragePairWiseDistanceMetric(proper_game),
            AveragePairWiseDistanceMetricOnlyPlayable(proper_game),
        ], log_to_wandb=True)
        experiment.do_all()
        wandb.finish()

        return f"Completed index = {index} with seed = {seed}"

    def one_run(index,  **kwargs):
        name = 'experiment_201_b'
        game = 'Mario'
        method = 'DirectGA'
        generations = kwargs['num_gens']
        keys = ['pop_size', 'num_gens', 'desired_entropy', 'desired_sparseness_enemies', 'desired_sparseness_coins', 'desired_sparseness_blocks',
                'entropy_block_size', 'enemies_block_size', 'coins_block_size', 'blocks_block_size', 'ground_maximum_height', 'coin_maximum_height', ]
        print(
            f"Doing now, with params = {' | '.join([f'{k}:{v}' for k, v in kwargs.items()])}")
        results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{"/".join([str(kwargs[key]) for key in keys])}'
        print(f"Results directory = {results_directory}")

        all_dic_args = dict(pop_size=kwargs['pop_size'],
                            number_of_generations=kwargs['num_gens'],
                            desired_entropy=kwargs['desired_entropy'],
                            desired_sparseness_enemies=kwargs['desired_sparseness_enemies'],
                            desired_sparseness_coins=kwargs['desired_sparseness_coins'],
                            desired_sparseness_blocks=kwargs['desired_sparseness_blocks'],
                            entropy_block_size=kwargs['entropy_block_size'],
                            enemies_block_size=kwargs['enemies_block_size'],
                            coins_block_size=kwargs['coins_block_size'],
                            blocks_block_size=kwargs['blocks_block_size'],
                            ground_maximum_height=kwargs['ground_maximum_height'],
                            coin_maximum_height=kwargs['coin_maximum_height'],)
        args = {
            'number_of_generations': generations,
            'level_gen': 'DirectGA',
            **all_dic_args
        }

        def get_pop():
            level = MarioLevel()
            game = MarioGame(level)
            return MarioGAPCG(game, level,
                              **all_dic_args
                              )

        futures = [single_func.remote(
            seed=i, index=index, name=name, game=game, method=method, results_directory=results_directory, args=args, date=date, get_pop=get_pop
        ) for i in range(5)]
        # We return the ray futures, and then only get them later, so that ray can run more than 5 at a time.
        return futures

    # The grid search thing

    counter = 0
    all_futures = []
    for pop_size in [20]:
        for num_gens in [100]:
            counter += 1
            all_futures += one_run(counter,
                                **{
                                    'pop_size': pop_size,
                                    'num_gens': num_gens,
                                    'desired_entropy': 0.0,
                                    'desired_sparseness_enemies': 0.0,
                                    'desired_sparseness_coins': 1.0,
                                    'desired_sparseness_blocks': 0.5,

                                    'entropy_block_size': 114,
                                    'enemies_block_size': 20,
                                    'coins_block_size': 10,
                                    'blocks_block_size': 10,

                                    'ground_maximum_height': 2,
                                    'coin_maximum_height': 2
                                }
                                )
    print(f"At end, we have {counter} runs to do. Running experiment_201_b")
    ray.get(all_futures)
    print("Done")

def experiment_201_d():
    """
    Runs the best of 201 c
    on batch
    """
    date = get_date()
    ray.init(num_cpus=14)

    @ray.remote
    def single_func(seed, index, name, game, method, results_directory, args, date, get_pop):
        # A single function to run one seed and one param combination.
        config = Config(
            name=name,
            game=game,
            method=method,
            seed=seed,
            results_directory=os.path.join(results_directory, str(seed)),
            method_parameters=args,
            date=date,
            project='NoveltyNeatPCG-Experiment201-d'
        )
        print("Date = ", config.date, config.results_directory,
              config.hash(seed=False))
        proper_game = MarioGame(MarioLevel(), do_enemies=True)
        parent = AStarDiversityAndDifficultyMetric(proper_game, number_of_times_to_do_evaluation=5)
        experiment = Experiment(config, get_pop, [
            CompressionDistanceMetric(proper_game),
            CompressionDistanceMetric(proper_game, use_combined_features=True),
            CompressionDistanceMetric(proper_game, do_mario_flat=True),
            LinearityMetric(proper_game),
            LeniencyMetric(proper_game),
            AStarSolvabilityMetric(proper_game, parent),
            AStarDifficultyMetric(proper_game, parent),
            AStarDiversityMetric(proper_game, parent),
            AStarEditDistanceDiversityMetric(proper_game, parent),

        ], log_to_wandb=True)
        experiment.do_all()
        wandb.finish()

        return f"Completed index = {index} with seed = {seed}"

    def one_run(index,  **kwargs):
        name = 'experiment_201_d'
        game = 'Mario'
        method = 'DirectGA'
        generations = kwargs['num_gens']
        keys = ['pop_size', 'num_gens', 'desired_entropy', 'desired_sparseness_enemies', 'desired_sparseness_coins', 'desired_sparseness_blocks',
                'entropy_block_size', 'enemies_block_size', 'coins_block_size', 'blocks_block_size', 'ground_maximum_height', 'coin_maximum_height']
        print(
            f"Doing now, with params = {' | '.join([f'{k}:{v}' for k, v in kwargs.items()])}")
        results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{"/".join([str(kwargs[key]) for key in keys])}' + "/use_novelty"
        print(f"Results directory = {results_directory}")

        all_dic_args = dict(pop_size=kwargs['pop_size'],
                            number_of_generations=kwargs['num_gens'],
                            desired_entropy=kwargs['desired_entropy'],
                            desired_sparseness_enemies=kwargs['desired_sparseness_enemies'],
                            desired_sparseness_coins=kwargs['desired_sparseness_coins'],
                            desired_sparseness_blocks=kwargs['desired_sparseness_blocks'],
                            entropy_block_size=kwargs['entropy_block_size'],
                            enemies_block_size=kwargs['enemies_block_size'],
                            coins_block_size=kwargs['coins_block_size'],
                            blocks_block_size=kwargs['blocks_block_size'],
                            ground_maximum_height=kwargs['ground_maximum_height'],
                            coin_maximum_height=kwargs['coin_maximum_height'], 
                            use_novelty=True)
        args = {
            'number_of_generations': generations,
            'level_gen': 'DirectGA',
            **all_dic_args
        }

        def get_pop():
            level = MarioLevel()
            game = MarioGame(level)
            return MarioGAPCG(game, level,
                              **all_dic_args
                              )

        futures = [single_func.remote(
            seed=i, index=index, name=name, game=game, method=method, results_directory=results_directory, args=args, date=date, get_pop=get_pop
        ) for i in range(5)]
        # We return the ray futures, and then only get them later, so that ray can run more than 5 at a time.
        return futures

    # The grid search thing

    counter = 0
    all_futures = []
    confs = [
        {
        'blocks_block_size': 10,
        'coin_maximum_height': 2,
        'coins_block_size': 10,
        'desired_entropy': 0.0,
        'desired_sparseness_blocks': 0.5,
        'desired_sparseness_coins': 1.0,
        'desired_sparseness_enemies': 0.0,
        'enemies_block_size': 20,
        'entropy_block_size': 114,
        'ground_maximum_height': 2,
        'level_gen': 'DirectGA',
        'number_of_generations': 100,
        'num_gens': 100,
        'pop_size': 100,
        'use_novelty': True,
        'index': 0
        }
    ]

    print(f"At end, we have {counter} runs to do. Running experiment_201_d")

    all_futures += one_run(**confs[0])
    print(f"WE have {len(all_futures)} futures to run")
    ray.get(all_futures)

if __name__ == '__main__':
    I = int(sys.argv[-1])
    if I == 0:
        print("RUNNING 201 a - DirectGA+ for Mario")
        experiment_201_a()
    elif I == 1:
        print("RUNNING 201b - DirectGA (Default) for Mario")
        experiment_201_b()
    elif I == 2:
        print("RUNNING 201d - DirectGA (Novelty) for Mario")
        experiment_201_d()
    else:
        assert False, f"Unsupported option {I}"
