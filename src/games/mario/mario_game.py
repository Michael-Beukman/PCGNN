from typing import Tuple

import numpy as np
from games.game import Game
from games.level import Level
from games.mario.mario_level import MarioLevel, MarioState


class MarioGame(Game):
    def __init__(self, level: MarioLevel, do_enemies: bool = True):
        super().__init__(level)
        self.mario_state = MarioState(do_enemies)
        self.mario_state.stringInitialize(level.string_representation_of_level(do_enemies).split("\n"))
        self.do_enemies = do_enemies
    
    def step(self, action: int) -> Tuple[bool, float]:
        x = action // 3 - 1
        y = action % 3 - 1
        self.mario_state.update(x, y)
        reward = -1
        
        if self.mario_state.checkLose():
            reward = -1000
        elif self.mario_state.checkWin():
            reward = 100
        
        self.current_pos = np.array([self.mario_state.player['x'], self.mario_state.player['y']])
        return self.mario_state.checkOver(), reward
    
    def reset(self, level: MarioLevel):
        super().reset(level)
        self.mario_state.stringInitialize(level.string_representation_of_level(self.do_enemies).split("\n"))
        self.current_pos = np.array([self.mario_state.player['x'], self.mario_state.player['y']])
    
    