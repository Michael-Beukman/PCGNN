from typing import List

from matplotlib import pyplot as plt
from games.game import Game
from games.level import Level
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.metric import Metric
from novelty_neat.maze.utils import path_length


class PathLengthMetric(Metric):
    """This is a simple metric, where we evaluate the difficulty using the length of the optimal path.
         Idea of using path length as a proxy for difficulty gotten from: Kegel and Haahr (2020):
          Barbara De Kegel and Mads Haahr. Procedural puzzle generation: A survey. IEEE Trans. Games, 12(1):21–40, 2020.
    """
    def __init__(self, game: Game) -> None:
        super().__init__(game)
    
    def evaluate(self, levels: List[Level]) -> List[float]:
        if (isinstance(self.game, MazeGame)):
            return self.evaluate_maze(levels)
        return super().evaluate(levels)
    
    def evaluate_maze(self, levels: List[Level]) -> List[float]:
        """
        Calculates the length of the shortest path of each level, normalised by the number of tiles.

        Args:
            levels (List[Level]): Levels to evaluate

        Returns:
            List[float]:
        """
        answer = []
        for level in levels:
            # How many tiles.
            norm = level.map.size
            answer.append(path_length(level) / norm)
        return answer
