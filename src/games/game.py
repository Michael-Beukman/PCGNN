from typing import Tuple
import numpy as np
from games.level import Level
class Game:
    """
        A game can be played (by a human/agent/etc).
        It requires a level and has some rules.
    """
    def __init__(self, level: Level):
        self.level = level
        self.current_pos = np.array([0, 0])

    def step(self, action: int) -> Tuple[bool, float]:
        """Should Step in the environment given the action.
        Args:
            action int
        Returns:
            done, reward
        """
        raise NotImplementedError()
    def reset(self, level: Level):
        """Resets this env given the level. The player now starts at the original spot again.

        Args:
            level (Level): The level to use.
        """
        self.level = level
        self.current_pos = np.array([0, 0])