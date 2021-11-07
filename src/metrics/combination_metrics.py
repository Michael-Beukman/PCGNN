from typing import List
from games.game import Game
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.metric import CombinationMetric
from metrics.rl.tabular.rl_agent_metric import RLAgentMetric
from metrics.solvability import SolvabilityMetric


class RLAndSolvabilityMetric(CombinationMetric):
    def __init__(self, game: Game, weights: List[float] = [1, 1]) -> None:
        super().__init__(game, metrics = [SolvabilityMetric(game), RLAgentMetric(game)], weights=weights)

class SolvabilityAndCompressionDistanceMetric(CombinationMetric):
    def __init__(self, game: Game, weights: List[float] = [1, 1]) -> None:
        super().__init__(game, metrics = [SolvabilityMetric(game), CompressionDistanceMetric(game)], weights=weights)