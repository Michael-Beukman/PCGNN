from typing import Tuple

import numpy as np
from games.game import Game
from games.level import Level


class MazeGame(Game):
    """
    Simple Maze Game.
    Some inspiration for rewards from here: https://www.samyzaf.com/ML/rl/qmaze.html
    """
    
    def reset(self, level: Level):
        """Resets this env given the level. The player now starts at the original spot again.

        Args:
            level (Level): The level to use.
        """
        self.level = level
        self.current_pos = level.start
        
    def __init__(self, level: Level):
        super().__init__(level)
    
    def step(self, action: int) -> Tuple[bool, float]:
        """Steps in the maze given the action and returns done, reward.

        Args:
            action (int): up = 0, right=1, down = 2, left = 3

        Returns:
            Tuple[bool, float]: done, reward
        """
        delta = np.array([0, 0])
        if action == 0:     delta[1] = -1
        elif action == 2:   delta[1] = 1
        elif action == 1:   delta[0] = 1
        elif action == 3:   delta[0] = -1
        new_pos = self.current_pos + delta
        x, y = new_pos
        is_bad = 0
        # if OOB or walking into wall, not allowed.
        if x < 0 or x >= self.level.width or y < 0 or y >= self.level.height or self.level.map[y, x] != 0:
            new_pos = self.current_pos
            is_bad = 1
        self.current_pos = new_pos
        # If at the end
        # if np.all(self.current_pos == np.array([self.level.width-1, self.level.height-1])):
        # Use the level's end position instead of (-1, -1)
        if np.all(self.current_pos == np.array(self.level.end)):
            return True, 1000
        else:
            # -1 living reward.
            return False, -1
        