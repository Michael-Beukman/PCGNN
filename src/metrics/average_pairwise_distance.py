from typing import List

import numpy as np
from games.game import Game
from games.level import Level
from games.maze.maze_game import MazeGame
from metrics.metric import Metric
from novelty_neat.novelty.distance_functions.distance import visual_diversity
import skimage.morphology as morph

class AveragePairWiseDistanceMetric(Metric):
    def __init__(self, game: Game) -> None:
        super().__init__(game)
    
    def evaluate(self, levels: List[Level]) -> List[float]:
        if (isinstance(self.game, MazeGame)):
            return self.evaluate_maze(levels)
        return super().evaluate(levels)
    
    def evaluate_maze(self, levels: List[Level]) -> List[float]:
        """
        Calculates the average pairwise distance between levels. Distance is the visual_diversity distance function.

        Args:
            levels (List[Level]): Levels to evaluate

        Returns:
            List[float]:
        """
        pairwise_dist = []
        for index1, l in enumerate(levels):
            # Connected components
            for level2 in levels[index1+1:]:
                pairwise_dist.append(visual_diversity(l.map, level2.map) / (l.map.shape[0] * l.map.shape[1]))

        return [float(np.mean(pairwise_dist))]



class AveragePairWiseDistanceMetricOnlyPlayable(Metric):
    def __init__(self, game: Game) -> None:
        super().__init__(game)
    
    def evaluate(self, levels: List[Level]) -> List[float]:
        if (isinstance(self.game, MazeGame)):
            return self.evaluate_maze(levels)
        return super().evaluate(levels)
    
    def evaluate_maze(self, levels: List[Level]) -> List[float]:
        """
        Calculates the average pairwise distance between levels. Distance is the visual_diversity distance function.
        The difference between this one and the AveragePairWiseDistanceMetric is that this metric first filters out any tiles that cannot be reached. 
            Thus, the distance will only be calculated using tiles that can actually be visited.

        Args:
            levels (List[Level]): Levels to evaluate

        Returns:
            List[float]:
        """
        pairwise_dist = []
        def clean(l: Level) -> np.ndarray:
            labelled = morph.label(l.map + 1, connectivity=1)
            labelled[labelled != labelled[0, 0]] = 0
            labelled[labelled == labelled[0, 0]] = 1
            return labelled
            
        cleaned_levels = [clean(l) for l in levels]

        for index1, l in enumerate(cleaned_levels):
            # Connected components
            for level2 in cleaned_levels[index1+1:]:
                pairwise_dist.append(visual_diversity(l, level2) / (l.shape[0] * l.shape[1]))

        return [float(np.mean(pairwise_dist))]


