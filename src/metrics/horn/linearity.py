from typing import List
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import numpy as np
import scipy
from games.game import Game
from games.level import Level
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.metric import Metric
from skimage import morphology as morph
class LinearityMetric(Metric):
    """
        This attempts to measure the linearity of the level. Specifically from Horn et al. (2014) and 
             G. Smith, J. Whitehead, M. Mateas, M. Treanor,J. March, and M. Cha. Launchpad: A rhythm-based level generator for 2-d platformers. Computational Intelligence and AI in Games, IEEE Transactions on, 3(1):1–16, 2011.
        
        There are two different implementations, depdending on the type of game.
    Args:
        Metric ([type]): [description]
    """
    def __init__(self, game: Game) -> None:
        super().__init__(game)
    
    def evaluate(self, levels: List[Level]) -> List[float]:
        if (isinstance(self.game, MazeGame)):
            return self.evaluate_maze(levels)
        if (isinstance(self.game, MarioGame)):
            return self.evaluate_mario(levels)
        return super().evaluate(levels)
    
    def evaluate_mario(self, levels: List[MarioLevel]) -> List[float]:
        """This runs the linearity metric on Mario. This basically takes the center point of each platform as a point, performs linear regression,
            and makes linearity the sum of the absolute difference between the actual position of a platform and the predicted one by the line.

        Args:
            levels (List[Level]): (Mario) Levels to Eval

        Returns:
            List[float]: The linearity for each level
        """
        ans = []
        for l in levels:
            # Find positions of solid objects
            num = l.tile_types_reversed['solid']
            where = np.argwhere(l.map == num)
            y, x = where[:, 0], where[:, 1]
            # https://stats.stackexchange.com/a/407568
            if len(x) < 2: 
                ans.append(0)
            else:
                model=LinearRegression()
                model.fit(x.reshape(-1,1), y)
                r2 = r2_score(y, model.predict(x.reshape(-1,1)))

                r_squared = r2
                linearity = r_squared
                ans.append(linearity)
        return ans

    def evaluate_maze(self, levels: List[Level]) -> List[float]:
        """
            Returns a list of all the linearity values.
            We basically choose the points that are connected to the start and end tiles,
            and compute the R2 value of this, to determine how linear this level is.
        Args:
            levels (List[Level]): Levels to eval.

        Returns:
            List[float]: Linearity per level
        """
        linearities = []
        for level in levels:
            connected = morph.label(level.map+1, connectivity=1)
            # Only select the tiles that are connected to the first one.
            connected[connected != connected[0, 0]] = 0
            coords = np.argwhere(connected)
            coords = coords.T
            if coords.shape[1] <= 2 or np.any(np.std(coords, axis=1) == 0):
                linearity = 0
            else:
                correlation_matrix = np.corrcoef(coords[0], coords[1])
                correlation_xy = correlation_matrix[0,1]
                r_squared = correlation_xy**2 
                linearity = r_squared
            linearities.append(linearity)
        return linearities

if __name__ == '__main__':
    def test_mario():
        level = MarioLevel()
        metric = LinearityMetric(MarioGame(level))
        level.map[10, 20:40] = 1
        ans = metric.evaluate([level])
        print(level.map)
        level.show()
        print("ANS = ", ans)
        pass
    
    test_mario();