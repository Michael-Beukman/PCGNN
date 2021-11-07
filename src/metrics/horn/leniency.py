from matplotlib import pyplot as plt
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
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

from novelty_neat.maze.utils import shortest_path

class LeniencyMetric(Metric):
    """The leniency metric.
     G. Smith, J. Whitehead, M. Mateas, M. Treanor, J. March, and M. Cha. Launchpad: A rhythm-based level generator for 2-d platformers. Computational Intelligence and AI in Games, IEEE Transactions on, 3(1):1–16, 2011.

    In general, the more ways there are the player can make mistakes, the less lenient the level is.
    """
    def __init__(self, game: Game) -> None:
        super().__init__(game)
    
    def evaluate(self, levels: List[Level]) -> List[float]:
        if (isinstance(self.game, MazeGame)):
            return self.evaluate_maze_better(levels)
        if (isinstance(self.game, MarioGame)):
            return self.evaluate_mario(levels)
        return super().evaluate(levels)

    def evaluate_mario(self, levels: List[Level]) -> List[float]:
        # Ok, so it seems that everyone's leniency is somewhat different
        # Horn et al has the most comparisons, but the least descriptive state of leniency
        # Shaker et al and Smith et al are more similar, and easy to compute.
        
        def single_level_value(level: MarioLevel):
            # find enemies
            # find all gaps
            # all jumps

            W, H = level.width, level.height
            map = level.map
            solid = level.tile_types_reversed['solid']
            hs = []
            for x in range(W):
                HH = np.argwhere(map[:, x] == solid)
                if HH.size == 0: 
                    HH = 0
                else:
                    # HH possibly has multiple different heights.
                    HH = np.max(HH)
                    HH = H - HH.squeeze()                
                h = HH
                hs.append(h)
            # Now we need to find the following using h:
            # 1. Gaps -> Where some heights are 0
            # 2. Jumps -> Where h(i+1) > h(i)
            gaps = []
            is_in_gap = False
            current_gap_width = 0
            number_of_jumps = 0
            
            is_in_flat = False
            current_flat_length = 0
            
            all_flats = []
            for i in range(1, len(hs)):
                prev = hs[i-1]
                now =  hs[i]

                if is_in_flat:
                    if now == prev and now != 0:
                        current_flat_length += 1
                    else:
                        if current_flat_length >= 3:
                            all_flats.append(current_flat_length)
                        current_flat_length = 0
                        is_in_flat = False
                else:
                    if now == prev and now != 0:
                        current_flat_length += 1
                        is_in_flat = True

                if now > prev and now != 0 and prev != 0:
                    number_of_jumps += 1
                if is_in_gap:
                    if now == 0:
                        current_gap_width += 1
                    else:
                        gaps.append(current_gap_width)
                        current_gap_width = 0
                        is_in_gap = False
                else:
                    # Not in gap
                    if now == 0:
                        is_in_gap = True
                        current_gap_width += 1
                    else:
                        pass
            if is_in_gap:
                gaps.append(current_gap_width)

            if is_in_flat:
                all_flats.append(current_flat_length)

            number_of_enemies = (map == level.tile_types_reversed['enemy']).sum()
            
            T = number_of_enemies + number_of_jumps + len(gaps)
            total_num_elements = max(1, T)
            total_leniency = number_of_enemies * -1 + number_of_jumps * 1 + len(gaps) * -1
            N = total_leniency / total_num_elements
            if T == 0: return 1.0
            else:
                return (N + 1) / 2
        
        return [single_level_value(l) for l in levels]
        

    def evaluate_maze_better(self, levels: List[Level]) -> List[float]:
        """This evaluates the leniency of a maze level in a slightly different way. We can measure leniency as the number of dead ends there are.
             Many dead ends indicate a difficult level, and few indicate an easy/lenient level.

             What we in effect do is the following, for each empty cell T, we block the shortest path from the start to T.
             If there is still a path from T to the goal, then that is lenient, otherwise not.
             We perform this for every empty tile that can be reached from the start.
        Args:
            levels (List[Level]): The levels to evaluate

        Returns:
            List[float]: The leniency metrics for each of them
        """
        final_answers = []
        for level in levels:
            labelled: np.ndarray = morph.label(level.map + 1, connectivity=1)
            # Only consider cells that are connected to the start.
            labelled[labelled != labelled[0, 0]] = 0
            labelled[labelled == labelled[0, 0]] = 1

            if labelled[0, 0] != labelled[-1, -1]:
                # If not solvable, then this doesn't make sense.
                final_answers.append(0)
                continue
            coords = np.argwhere(labelled == 1)
            non_blocked = 0
            # x, y
            goal = (labelled.shape[1] - 1, labelled.shape[0] - 1)
            for row, column in coords:
                if row == 0 and column == 0 or row == goal[1] and column == goal[0]:
                    # Don't focus on start and end.
                    non_blocked += 1
                    continue
                temp = labelled.copy()
                path = shortest_path(temp, (0, 0), (column, row), 0)
                assert path is not None
                # Now clear the path, excluding the last node.
                for x, y in path[:-1]:
                    temp[y, x] = 0
                path = shortest_path(temp, (column, row), goal, 0)
                temp[row, column] = 5
                if path is not None: 
                    non_blocked += 1
            
            final_answers.append(non_blocked / labelled.sum())
        return final_answers
