from typing import List, Tuple, Union

import numpy as np
from games.game import Game
from games.level import Level
from games.mario.java_runner import java_astar_number_of_things_in_open_set
from games.mario.mario_game import MarioGame
from games.maze.maze_game import MazeGame
from metrics.metric import Metric
from metrics.rl.tabular.rl_agent_metric import compare_actions_edit_distance, sampled_norm_trajectory_comparison
from novelty_neat.maze.a_star import do_astar_from_level
Actions = List[int]
Trajectory = List[Tuple[int, int]]

AllActionsAndTrajectories = Tuple[List[Trajectory], List[Actions]]


class AStarDiversityAndDifficultyMetric(Metric):
    """
    This metric does the following
        Run an A* agent on the level (Maze or Mario), and compares the trajectories
        Using the sampled normalised distance metric or edit distance on the actions.

    This also calculates the difficulty of each level by finding out how many nodes A* expanded.

    It can also keep track of solvability.
    """

    def __init__(self, game: Game, use_edit_distance=False, n_samples: Union[int, None] = None, 
                 number_of_times_to_do_evaluation: int = 1) -> None:
        """
        Args:
            game (Game): The game
            use_edit_distance (bool, optional): Whether or not to use edit distance. Defaults to False.
            n_samples (int, optional): Defaults to 30.
            number_of_times_to_do_evaluation (int, optional): If this is greater than 1, 
                we evaluate multiple times and return metric values that are the combination of the multiple values
                Specifically, solvability is the maximum
                              difficulty is the mean
                              diversity is the mean
                Defaults to 1.
        """
        super().__init__(game)
        self.use_edit_distance: bool = use_edit_distance
        self.action_trajectories: Union[bool, AllActionsAndTrajectories] = None
        if n_samples is None:
            self.n_samples = game.level.width * 2
            if game.level.width == 14: self.n_samples = 30
        else:    
            self.n_samples = n_samples
        self.number_of_times_to_do_evaluation = number_of_times_to_do_evaluation

        self.difficulty: List[float] = []
        self.diversity: List[float] = []
        self.solvability: List[float] = []
        self.edit_distance_diversity: List[float] = []

        self.all_difficulty: List[List[float]] = []
        self.all_diversity:  List[List[float]] = []
        self.all_edit_distance_diversity:  List[List[float]] = []
        self.all_solvability: List[List[float]] = []
        self.all_trajs = []

    def evaluate(self, levels: List[Level], action_trajectories: Union[None, AllActionsAndTrajectories] = None) -> List[float]:
        if self.number_of_times_to_do_evaluation > 1:
            assert action_trajectories is None, "Action trajectories cannot be given if we evaluate multiple times"
            
            # evaluate multiple times because the results are not always equal.
            for i in range(self.number_of_times_to_do_evaluation):
                self._evaluate(levels)
                self.all_difficulty.append(self.difficulty)
                self.all_diversity.append(self.diversity)
                self.all_solvability.append(self.solvability)
                self.all_trajs.append(self.action_trajectories)
                self.all_edit_distance_diversity.append(self.edit_distance_diversity)
            
            # Here we combine these now
            self.difficulty   = np.mean(self.all_difficulty, axis=0)
            self.diversity    = np.mean(self.all_diversity, axis=0)
            self.solvability  = np.amax(self.all_solvability, axis=0)
            
            self.edit_distance_diversity  = np.mean(self.all_edit_distance_diversity, axis=0)
            return self.diversity

        return self._evaluate(levels, action_trajectories)

    def _evaluate(self, levels: List[Level], action_trajectories: Union[None, AllActionsAndTrajectories] = None) -> List[float]:
        if action_trajectories is None:
            self.action_trajectories = self._get_action_trajectories(levels)
            action_trajectories = self.action_trajectories

        w, h = self.game.level.width, self.game.level.height
        if (isinstance(self.game, MarioGame)):
            w, h = self.game.mario_state.width, self.game.mario_state.height
            pass

        # use either actions or trajs.
        for edit_distance in [True, False]:
            overall_dist = []
            things_to_compare = action_trajectories[1] if edit_distance else action_trajectories[0]
            comparison_func = compare_actions_edit_distance if edit_distance else sampled_norm_trajectory_comparison
            # For all pairs
            for i in range(len(things_to_compare)):
                for j in range(len(things_to_compare))[i+1:]:
                    x = things_to_compare[i]
                    y = things_to_compare[j]
                    # Compare and append to output array.
                    d = comparison_func(x, y, w, h, self.n_samples)
                    overall_dist.append(d)
            if edit_distance:
                self.edit_distance_diversity = overall_dist
            else:
                self.diversity = overall_dist
        if self.use_edit_distance:
            return self.edit_distance_diversity
        else:
            return self.diversity

    def _get_action_trajectories(self, levels: List[Level]) -> AllActionsAndTrajectories:
        """Returns a tuple of trajectories and actions. It also sets the difficulty.

        Args:
            levels (List[Level]): 

        Returns:
            AllActionsAndTrajectories: Trajs, Action_Trajs
        """
        trajs = []
        actions = []
        difficulty = []
        solvs = []
        if isinstance(self.game, MarioGame):
            time_to_take = 50
            if levels[0].width > 114:
                time_to_take += (levels[0].width / 114) * 50
                time_to_take = int(time_to_take)

            for level in levels:
                solved, traj, action_traj, diff = java_astar_number_of_things_in_open_set(
                    level, time_per_episode=time_to_take)
                trajs.append(traj)
                actions.append(action_traj)
                difficulty.append(diff)
                solvs.append(1.0 if solved else 0.0)

        elif isinstance(self.game, MazeGame):
            for level in levels:
                path, sets, num_considered = do_astar_from_level(level)
                is_solvable = True
                if path is None or path == -1 or len(path) == 0:
                    path = [(0, 0) for _ in range(self.n_samples)]
                    is_solvable = False
                action_traj = self._get_actions_from_trajectory(path)

                trajs.append(path)
                actions.append(action_traj)

                num_passable_tiles = max(
                    1, (level.map == level.tile_types_reversed['empty']).sum() - len(path))
                difficulty.append(max(num_considered - len(path), 0) / num_passable_tiles)
                solvs.append(is_solvable)
        else:
            raise Exception("Invalid Game provided")
        self.difficulty = difficulty
        self.solvability = solvs
        return trajs, actions

    def _get_actions_from_trajectory(self, traj: Trajectory) -> Actions:
        """Returns the actions made to get to this trajectory, specifically
             each action is a number from 0 - 9, indicating where in the moore neighbourhood the agent
             moved to, or if it stayed stationary.

        Args:
            traj (Trajectory): The trajectory

        Returns:
            Actions: List of individual actions
        """
        ans = []
        if len(traj) >= 2:
            curr = traj[0]
            for next_one in traj[1:]:
                diff = (curr[0] - next_one[0], curr[1] - next_one[1])
                assert -1 <= diff[0] <= 1
                assert -1 <= diff[1] <= 1

                # transforms it to 0, 1 and 2.
                diff = (
                    2 if diff[0] == -1 else diff[0],
                    2 if diff[1] == -1 else diff[1],
                )

                action = diff[0] * 3 + diff[1]
                ans.append(action)
                curr = next_one

        if len(ans) == 0:
            ans.append(0)
        return ans


class AStarDiversityMetric(Metric):
    """This simply computes the diversity using the above AStarDiversityAndDifficultyMetric
    """

    def __init__(self, game: Game, parent: AStarDiversityAndDifficultyMetric) -> None:
        super().__init__(game)
        self.parent = parent

    def evaluate(self, levels: List[Level]) -> List[float]:
        if len(self.parent.diversity) == 0:
            return self.parent.evaluate(levels)
        else:
            return self.parent.diversity


class AStarEditDistanceDiversityMetric(Metric):
    """This simply computes the diversity using edit distance of trajectories using the above AStarDiversityAndDifficultyMetric
    """

    def __init__(self, game: Game, parent: AStarDiversityAndDifficultyMetric) -> None:
        super().__init__(game)
        self.parent = parent

    def evaluate(self, levels: List[Level]) -> List[float]:
        if len(self.parent.edit_distance_diversity) == 0:
            self.parent.evaluate(levels)
        return self.parent.edit_distance_diversity



class AStarDifficultyMetric(Metric):
    """This simply computes the difficulty using the above AStarDiversityAndDifficultyMetric
    """

    def __init__(self, game: Game, parent: AStarDiversityAndDifficultyMetric) -> None:
        super().__init__(game)
        self.parent = parent

    def evaluate(self, levels: List[Level]) -> List[float]:
        if len(self.parent.difficulty) == 0:
            self.parent.evaluate(levels)
        return self.parent.difficulty


class AStarSolvabilityMetric(Metric):
    """This simply computes the solvability using the above AStarDiversityAndDifficultyMetric
    """

    def __init__(self, game: Game, parent: AStarDiversityAndDifficultyMetric) -> None:
        super().__init__(game)
        self.parent = parent

    def evaluate(self, levels: List[Level]) -> List[float]:
        if len(self.parent.solvability) == 0:
            self.parent.evaluate(levels)
        return self.parent.solvability
    
    def name(self):
        return "SolvabilityMetric"
