from typing import List
from games.level import Level
from games.game import Game
from metrics.metric import Metric
from Levenshtein import distance as levenshtein_distance

class EditDistanceMetric(Metric):
    """This simply calculates the edit distance between levels by flattening the map.
    """

    def __init__(self, game: Game) -> None:
        super().__init__(game)

    def evaluate(self, levels: List[Level]) -> List[float]:
        ans = []
        string_levels = [''.join(map(str, level.map.flatten())) for level in levels]
        for i in range(len(string_levels)):
            for j in range(len(string_levels))[i+1:]:
                sa = string_levels[i]
                sb = string_levels[j]
                ans.append(levenshtein_distance(sa, sb) / max(len(sa), len(sb)))

        return ans

class HammingDistanceMetric(Metric):
    """This calculates the pairwise Hamming distance between the levels.
    """
    def __init__(self, game: Game) -> None:
        super().__init__(game)
    def evaluate(self, levels: List[Level]) -> List[float]:
        ans = []
        for i in range(len(levels)):
            for j in range(len(levels))[i+1:]:
                a = levels[i].map
                b = levels[j].map
                # Number of unequal tiles, normalised to between 0 and 1.
                ans.append((a != b).sum() / b.size)

        return ans