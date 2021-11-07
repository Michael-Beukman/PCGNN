from matplotlib import pyplot as plt
from games.mario.assets.engine import MarioAstarAgent
from games.mario.java_runner import java_solvability
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
from gym_pcgrl.envs.probs.smb.engine import State
from metrics.metric import Metric
from pydoc import plain
from typing import List

import numpy as np
from games.game import Game
from games.level import Level
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.metric import Metric
from skimage import morphology as morph
from scipy import signal

class SolvabilityMetric(Metric):
    """
    """
    def __init__(self, game: Game) -> None:
        super().__init__(game)
    
    def evaluate(self, levels: List[Level]) -> List[float]:
        if (isinstance(self.game, MazeGame)):
            return self.evaluate_maze(levels)
        elif (isinstance(self.game, MarioGame)):
            return self.evaluate_mario(levels)
        return super().evaluate(levels)
    
    def evaluate_mario(self, levels: List[Level]) -> List[float]:
        ans = []
        for level in levels:
            ans.append(java_solvability(level) * 1.0)
        return ans
        def single_level_solv(level: MarioLevel):
            # lvlString = level.string_representation_of_level()
            
            # In this case make enemies look like solid tiles, to get around the very hard enemy placement.
            lvlString_enemies = level.string_representation_of_level(True).replace("x", '#')
            lvlString_not_enemies = level.string_representation_of_level(False)
            def get_solv_of_string(lvlString):
                state = State()
                state.stringInitialize(lvlString.split("\n"))

                aStarAgent = MarioAstarAgent()

                sol,solState,iters = aStarAgent.getSolution(state, 1, 10000)
                if solState.checkWin():
                    return 1
                sol,solState,iters = aStarAgent.getSolution(state, 0, 10000)
                if solState.checkWin():
                    return 1
                return 0
            
            return get_solv_of_string(lvlString_enemies) == 1 and get_solv_of_string(lvlString_not_enemies) == 1

        return [single_level_solv(l) for l in levels]

    def evaluate_maze(self, levels: List[Level]) -> List[float]:
        """
        Since this is for the maze game, the solvability metric is simply 1 if there is a path between the start and end, and 0 otherwise.

        Args:
            levels (List[Level]): Levels to evaluate

        Returns:
            List[float]: [description]
        """
        solvability = []
        for l in levels:
            # Connected components
            connected = morph.label(l.map+1, connectivity=1)
            if connected[0, 0] == connected[-1, -1] and self.game.level.tile_types[int(l.map[0, 0])] == 'empty':
                solvability.append(1)
            else:
                solvability.append(0)

        return solvability


if __name__ == '__main__':
    def test_mario_solv():
        game = MarioGame(MarioLevel())
        level_solv = MarioLevel()
        level_not_solv = MarioLevel()
        level_not_solv.map[-1, 10:30] = 0
        solv_metric = SolvabilityMetric(game)
        print("Level 1 solv: ", solv_metric.evaluate([level_solv]))
        
        print("Level 2 solv: ", solv_metric.evaluate([level_not_solv]))
        level_not_solv.show(); plt.show()
        
        pass
    test_mario_solv(); exit()