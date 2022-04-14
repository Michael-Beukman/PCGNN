import os
from typing import Any, Callable, Dict, List
from attr import has

import numpy as np
from common.methods.pcg_method import PCGMethod
from common.types import Verbosity
from common.utils import clean_name, get_date
from experiments.results import Results
from experiments.config import Config
from experiments.logger import WandBLogger, Logger, WriteToDictionaryLogger
from metrics.metric import Metric
from timeit import default_timer as tmr
import random

class Experiment:
    # Number of levels to generate on to evaluate
    NUM_LEVELS = 100

    """
        This should run a single experiment, including training, evaluating and storing results.
    """
    def __init__(self, config: Config, get_method: Callable[[], PCGMethod], metrics: List[Metric], log_to_wandb=True, verbose: Verbosity = Verbosity.NONE, log_to_dict: bool = False):
        """
            Initialises this experiment.
        Args:
            config (Config): The configuration, containing the parameters, seed, etc.
            get_method (Callable[[], PCGMethod]): A simple function that returns a PCGMethod. 
                This is because we seed before we create the PCGMethod, so any creation is reproducible.
            metrics (List[Metric]): The list of metrics to use for evaluation.
            log_to_wandb (bool): If this is true, then we create a wandb logger. Otherwise, we simply make a standard, no-op logger. 
                Useful for testing some things.
            verbose (Verbosity): How verbose should the method be.
            log_to_dict (bool): If true, logs all fitnesses to a pickle file.
        """
        self.config = config
        np.random.seed(config.seed)
        random.seed(config.seed)
        self.method = get_method()
        if log_to_dict:
            self.logger: Logger = WriteToDictionaryLogger(config, verbose, seed=config.seed)
        elif log_to_wandb:
            self.logger: Logger = WandBLogger(config, verbose, seed=config.seed)
        else:
            self.logger: Logger = Logger(config, verbose, seed=config.seed)
        # self.logger = Logger(config)
        self.metrics = metrics
    
        self.train_results: List[Dict[str, Any]] = None
        
        self.eval_results_single: Dict[str, float] = {}
        self.eval_results_all: Dict[str, List[float]] = {}
        self.train_time = 0
        self.generation_time = 0
        self.levels_to_eval = []
        self.extra_results = []

        self.extra_information: Dict[str, Dict[str, Any]] = {}

    def train(self):
        start = tmr()
        self.train_results = self.method.train(self.logger)
        end = tmr()
        self.train_time = end - start
        self.logger.log({'train_time': self.train_time})
        self.logger.end(self.method.game.__class__.__name__, self.config.seed)

    def evaluate(self, step: int = 0):
        """
            Evaluates the generator on the specified metrics.
        """
        start = tmr()
        self.levels_to_eval = []
        for i in range(Experiment.NUM_LEVELS):
            self.levels_to_eval.append(self.method.generate_level())
        
        print("Evaluating levels now")
        end = tmr()
        self.generation_time = (end - start) / Experiment.NUM_LEVELS # average time to generate single level.
        for metric in self.metrics:
            values = metric.evaluate(self.levels_to_eval)
            self.eval_results_single[metric.name()] = np.mean(values)
            self.eval_results_all[metric.name()] = values
            try:
                if hasattr(metric, 'action_trajectories'):
                    self.extra_results.append({
                        'action_trajectories': metric.action_trajectories
                    })
            except Exception as e:
                print("Failed when adding action trajs, ", e)
            self.extra_information[metric.name()] = metric.useful_information()
        
        self.logger.log({'generation_time': self.generation_time, 'evaluation_results': self.eval_results_single}, step=step)
        print("Done evaluating")

    def save_results(self):
        """
            Save results to file.
        """
        results = Results(self.config, self.train_results, self.eval_results_single, self.eval_results_all, self.train_time, self.generation_time, levels=self.levels_to_eval, extra_results=self.extra_information, metrics=self.metrics)
        filename=  f"seed_{self.config.seed}_name_{clean_name(self.config.name)}_{get_date()}.p"
        results.to_file(self.config.results_directory, filename)
        print(f"Saving results to {os.path.join(self.config.results_directory, filename)}")
    
    def do_all(self, step = 0):
        self.train()
        self.evaluate(step + 9999)
        self.save_results()
        return self.eval_results_single