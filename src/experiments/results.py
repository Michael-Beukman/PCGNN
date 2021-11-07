import os
import pickle
from typing import Any, Dict, List

from games.level import Level
from experiments.config import Config
from metrics.metric import Metric


class Results:
    def __init__(self, 
                config: Config,
                train_results: List[Dict[str, Any]],
                 eval_results_single: Dict[str, float],
                 eval_results_all: Dict[str, List[float]],
                 train_time: float,
                 generation_time: float,
                 levels: List[Level],
                 extra_results: Any,
                 metrics: List[Metric]
                 ) -> None:
        """

        Args:
            train_results (List[Dict[str, Any]]): The training data. List of datapoints.
            eval_results_single (Dict[str, float]): Evaluation data, { metric_name: average_value }
            eval_results_all (Dict[str, List[float]]): Evaluation data in list form, { metric_name: list_of_values }
            train_time float: Time (s) taken to train method
            generation_time float: Time (s) taken to generate one level
        """
        self.train_results = train_results
        self.eval_results_single = eval_results_single
        self.eval_results_all = eval_results_all
        self.train_time = train_time
        self.generation_time = generation_time
        self.levels = levels
        self.config = config
        self.extra_results = extra_results
        self.metrics = metrics

    def to_file(self, directory_name: str, filename: str):
        os.makedirs(directory_name, exist_ok=True)
        with open(os.path.join(directory_name, filename), 'wb+') as f:
            pickle.dump({
                'config': self.config,
                'train_results': self.train_results,
                'eval_results_single': self.eval_results_single,
                'eval_results_all': self.eval_results_all,
                'train_time': self.train_time,
                'generation_time': self.generation_time,
                'levels': self.levels,
                'extra_results': self.extra_results,
                'metrics': self.metrics
            }, f)
