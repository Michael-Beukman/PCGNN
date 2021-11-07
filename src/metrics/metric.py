from typing import Any, Dict, List

import numpy as np
from games.game import Game

from games.level import Level


class Metric:
    """
        A metric is a way to evaluate levels.
    """

    def __init__(self, game: Game) -> None:
        self.game = game
        pass

    def evaluate(self, levels: List[Level]) -> List[float]:
        """
            Evaluates this collection of levels and returns a list of floats.
                Averaging this list should give an accurate indication of the average value of this metric.
            For example, when measuring diversity using the compression distance, the list could be of length 
                len(levels) * len(levels) indicating the diversity of each pair of levels.
                When measuring the difficulty of a level, this list could simply be the difficulty of each level.

        Args:
            levels (List[Level]): The levels to evaluate

        Returns:
            List[float]: The evaluation scores.
        """
        return [0]
    @classmethod
    def name(cls):
        """
            Returns a name of this metric
        """
        return str(cls.__name__)

    def useful_information(self) -> Dict[str, Any]:
        """This returns any useful information from the metric, 
            after the evaluate function has been called.
        Returns:
            Dict[str, Any]: Any information.
        """
        return {}

class CombinationMetric(Metric):
    def __init__(self, game: Game, metrics: List[Metric], weights: List[float] = None) -> None:
        super().__init__(game)
        self.metrics = metrics
        if weights is None:
            weights = np.ones(len(metrics))
            if weights.sum() == 0:
                weights[0] = 1
            
        self.weights = weights / np.sum(weights)
        assert len(self.metrics) != 0, "Must have some metrics"
    
    def evaluate(self, levels: List[Level]) -> List[float]:
        score = 0.0
        for metric, w in zip(self.metrics, self.weights):
            score += w * np.mean(metric.evaluate(levels))
        return [score]