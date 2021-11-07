from math import floor
import os
import pickle
import pstats
import random
from typing import Any, Callable, Dict, List, Tuple
from matplotlib import animation, pyplot as plt
import cProfile
import numpy as np
import scipy
import scipy.stats
from common.utils import get_date
from experiments.config import Config
from experiments.logger import Logger, NoneLogger, WandBLogger
from games.game import Game
from games.gym_wrapper import GymMarioWrapper, GymMazeWrapper, LevelGenerator
from games.level import Level
from games.mario.java_runner import java_rl_diversity
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel

from metrics.metric import Metric
from metrics.rl.tabular.tabular_rl_agent import TabularRLAgent
import scipy.spatial
from Levenshtein import distance as levenshtein_distance


_Position = Tuple[int, int]
"""A Trajectory is simply a list of (x, y) coordinates representing where the agent was at each timestep.
    This is used extensively in the following file
"""
_Trajectory = List[_Position]

class _FixedLevelGenerator(LevelGenerator):
    def __init__(self, level: Level):
        super().__init__()
        self.level = level

    def get_level(self) -> Level:
        return self.level

# Trajectory comparison, takes in two trajectories, as well as the width and height, as well as the size of the sampling trajectory of a level and returns a float representing their distance.
# Should be normalised to between 0 and 1, where larger values mean that the two trajectories are different, and smaller values mean that the
# trajectories are similar.

TrajectoryComparison = Callable[[_Trajectory, _Trajectory, int, int, int], float]

def jensen_shannon_divergence_trajectory_comparison(a: _Trajectory, b: _Trajectory, w: int, h: int, n_trajectory_samples: int = 30) -> float:
    """This computes the Jensen Shannon divergence from these two trajectories. It creates a probability distribution from it, and returns the divergence.
        
        Sources:
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html

    Args:
        a (_Trajectory): The first trajectory
        b (_Trajectory): The second trajectory
        w (int): width of level
        h (int): height of level
        n_trajectory_samples (int, optional): Number of samples to generate uniformly along each trajectory. This is not used. 
            If needed, look at jensen_shannon_divergence_trajectory_comparison_sampled. Defaults to 30.

    Returns:
        float: The diversity from these two levels, between 0 and 1
    """
    p1 = make_probability_distribution_from_trajectory(a, w, h, False)
    p2 = make_probability_distribution_from_trajectory(b, w, h, False)
    return scipy.spatial.distance.jensenshannon(p1, p2, base=2)
   

def sampled_norm_trajectory_comparison(a: _Trajectory, b: _Trajectory, w: int, h: int, n_trajectory_samples: int = 30) -> float:
    """This first samples n_trajectory_samples from the trajectories, and then returns the value of simplified_norm_trajectory_comparison


    Args:
        a (_Trajectory): The first trajectory
        b (_Trajectory): The second trajectory
        w (int): width of level
        h (int): height of level
        n_trajectory_samples (int, optional): Number of samples to generate uniformly along each trajectory. This is not used. Defaults to 30.

    Returns:
        float: The diversity from these two levels, between 0 and 1
    """
    return simplified_norm_trajectory_comparison(
        sample_trajectory(a, n_trajectory_samples),
        sample_trajectory(b, n_trajectory_samples),
        w, h, n_trajectory_samples
    )
    pass


class RLAgentMetric(Metric):
    """
        A metric to evaluate levels based on the action trajectories of an RL agent.
    """
    def __init__(self, game: Game, trajectory_comparison: TrajectoryComparison = sampled_norm_trajectory_comparison, n_samples: int = 30, verbose: bool = False) -> None:
        """
        Args:
            game (Game): The game to evalute on.
            trajectory_comparison (TrajectoryComparison): This is a function that takes in two _Trajectories, 
                the width and height of the level, as well as the number of samples to take along a trajectory. 
                It returns a floating point value describing how different (maximally different = 1) or similar (maximum similarity = 0).
            n_samples (int, optional): The number of samples to take along the trajectory. Some trajectory_comparison functions use this
                to ensure that different trajectories can be made the same length. Defaults to 30.
        """
        super().__init__(game)
        self.comparison = trajectory_comparison
        self.n_samples = n_samples
        self.verbose = verbose
        self.action_trajectories = None
    
    def evaluate(self, levels: List[Level], action_trajectories: List[_Trajectory] = None) -> List[float]:
        """This evaluates the levels using the specified comparison function.

        Args:
            levels (List[Level]): The list of levels. Let n = len([l for l in levels where l.map[0, 0] != filled])
                                    Thus, if a level has a filled block in the first position, we ignore it.
                                    Other unsolvable levels are fair game.
            action_trajectories (List[_Trajectory]) If this is given, then we don't calculate the trajectories but rather use the provided ones.
        Returns:
            List[float]: List of length (n)(n+1)/2 detailing the distance between each pair of levels.
        """
        good_levels = [l for l in levels if l.map[0, 0] == 0]
        if len(good_levels) == 0:
            # All levels are bad
            return [0]
        levels = good_levels
        # Get the trajectories for each level.
        if action_trajectories is None:
            self.action_trajectories = self._get_action_trajectories(levels, True)
            action_trajectories = self.action_trajectories[0]
        
        overall_dist = []
        w, h = self.game.level.width, self.game.level.height
        if (isinstance(self.game, MarioGame)):
            w, h = self.game.mario_state.width, self.game.mario_state.height
            pass
        # For all pairs
        for i in range(len(action_trajectories)):
            for j in range(len(action_trajectories))[i+1:]:
                x = action_trajectories[i]
                y = action_trajectories[j]
                # Compare and append to output array.
                d = self.comparison(x, y, w, h, self.n_samples)
                overall_dist.append(d)
        return overall_dist
    

    def _get_action_trajectories(self, levels: List[Level], get_actions_too: bool=False) -> List[_Trajectory]:
        """Returns the action trajectories for these levels. There are of the form
             [(x0, y0), (x1, y1), ..., (xn, yn)], i.e. a list of the locations that the agent was at at each timestep.

        Args:
            levels (List[Level]): Levels
            get_actions_too (bool): If this is True, then we return the actions too instead of just the trajectories.

        Returns:
            List[_Trajectory]: The list of trajectories for each level.
        """
        if isinstance(self.game, MarioGame):
            return self._get_action_trajectories_mario(levels, get_actions_too)
        all_trajectories = []
        all_actions = []


        alpha = 1
        # timestep_cap = (self.game.level.width + self.game.level.height) * 10 # 100 * 10
        timestep_cap = 1000 # {14: 100, 30: 400, 100: 1000}.get(self.game.level.width, (self.game.level.width + self.game.level.height) * 10)
        # timestep_cap = (self.game.level.width + self.game.level.height) * 10 # 100 * 10
        num_states = lambda env, level: level.map.size
        eps = 5000
        if (isinstance(self.game, MarioGame)):
            alpha = 0.5
            timestep_cap = 500
            num_states = lambda env, level: env.observation_space.n
            eps = 2000
            # solved = lambda eval_rewards: all([e[-1] > -timestep_cap for e in eval_rewards])


        if 1 or (isinstance(self.game, MazeGame)):
            for level in levels:
                gen = _FixedLevelGenerator(level)
                if (isinstance(self.game, MazeGame)):
                    env = GymMazeWrapper(level_generator=gen, timestep_cap=timestep_cap, number_of_level_repeats=0, init_level=level)
                else:
                    env = GymMarioWrapper(level_generator=gen, timestep_cap=timestep_cap, number_of_level_repeats=0, init_level=level)
                agent = TabularRLAgent(num_states(env, level), env.action_space.n, alpha=alpha)
                rs = agent.train(env, episodes=eps, verbose=self.verbose)

                w = level.map.shape[0]
                if (isinstance(self.game, MarioGame)):
                    w = self.game.mario_state.width,
                if level.map[0, 0] == level.tile_types_reversed['filled']:
                    N = timestep_cap
                    T, A = [0 for _ in range(N)], [0 for _ in range(N)]
                else:
                    T, A = agent.get_trajectory(env, True)
                trajectory = [
                    (t % w, t // w) for t in T
                ]
                all_actions.append(A)
                all_trajectories.append(trajectory)
            if get_actions_too:
                return all_trajectories, all_actions
            return all_trajectories
        else:
            # agent = TabularRLAgent(level.map.size, env.action_space.n, alpha = 0.5)
            pass
    
    def _get_action_trajectories_mario(self, levels: List[MarioLevel], get_actions_too: bool=False) -> List[_Trajectory]:
        trajs = []
        actions = []
        for l in levels:
            solved, a, states, positions = java_rl_diversity(l)
            positions = np.array(positions) / 16
            positions = [tuple(p) for p in positions]
            trajs.append(positions)
            actions.append(a)
        if get_actions_too:
            return trajs, actions
        return trajs

    def __repr__(self) -> str:
        return f"RLAgentMetric({self.comparison.__name__}, n_samples={self.n_samples})"

    def useful_information(self) -> Dict[str, Any]:        
        if self.action_trajectories is None: return super().useful_information()
        return {
            'trajectories': self.action_trajectories[0],
            'action_trajectories': self.action_trajectories[1],
        }

def sample_trajectory(traj: _Trajectory, N: int) -> _Trajectory:
    """This takes the input trajectory and sample N points uniformly, returning a trajectory of length N.

    Args:
        traj (_Trajectory): 
        N (int): Number of points to sample

    Returns:
        _Trajectory: New, sampled trajectory of length N
    """
    ns = np.linspace(0, 1, N, endpoint=False)
    new_values = []
    for n in ns:
        new_values.append(traj[floor(n * len(traj))])
    return new_values

def make_probability_distribution_from_trajectory(traj: _Trajectory, width: int, height: int, normalise: bool = False) -> np.ndarray:
    """This takes in a trajectory, and creates a probability distribution from it by first linearising the index of each location,
        and counting how many times that location came up.
        It returns an array of length (width * height) representing the counts of each location.

    Args:
        traj (_Trajectory): The input trajectory
        width (int): Dimensions of level
        height (int): Dimensions of level
        normalise (bool, optional): If this is true, the probability distribution is normalised so that it sums to 1. Defaults to False.

    Returns:
        np.ndarray: The distribution.
    """
    prob = np.zeros(width * height)
    for x, y in traj:
        index = y * width + x
        prob[index] += 1
    if normalise:
        prob = prob / prob.sum()
    return prob
    
        

def jensen_shannon_divergence_trajectory_comparison_sampled(a: _Trajectory, b: _Trajectory, w: int, h: int, n_trajectory_samples: int = 30) -> float:
    """This simply samples n_trajectory_samples from the trajectories and then computes the Jensen-Shannon Divergence from it
        using jensen_shannon_divergence_trajectory_comparison

    Args:
        a (_Trajectory): The first trajectory
        b (_Trajectory): The second trajectory
        w (int): width of level
        h (int): height of level
        n_trajectory_samples (int, optional): Number of samples to generate uniformly along each trajectory. Defaults to 30.

    Returns:
        float: The diversity from these two levels, between 0 and 1
    """
    return jensen_shannon_divergence_trajectory_comparison(sample_trajectory(a, n_trajectory_samples), sample_trajectory(b, n_trajectory_samples),
            w, h, n_trajectory_samples)
 

def wasserstein_distance_trajectory_comparison(a: _Trajectory, b: _Trajectory, w: int, h: int, n_trajectory_samples: int = 30) -> float:
    """Computes the earth mover distance (EMD) from the two trajectories by making them into a probability distribution.

    Args:
        a (_Trajectory): The first trajectory
        b (_Trajectory): The second trajectory
        w (int): width of level
        h (int): height of level
        n_trajectory_samples (int, optional): Number of samples to generate uniformly along each trajectory. This is not used. Defaults to 30.

    Returns:
        float: The EMD between these two trajectories.
    """
    p1 = make_probability_distribution_from_trajectory(a, w, h, False)
    p2 = make_probability_distribution_from_trajectory(b, w, h, False)

    p1 = np.argwhere(p1)[:, 0]
    p2 = np.argwhere(p2)[:, 0]
    
    return scipy.stats.wasserstein_distance(p1, p2)
    

def l2_norm_trajectory_comparison(a: _Trajectory, b: _Trajectory, w: int, h: int, n_trajectory_samples: int = 30) -> float:
    """
        Computes probability distributions p and q from trajectories a and b, respectively, and then computes
            sum over i (p[i] - q[i]) ** 2

    Args:
    Args:
        a (_Trajectory): The first trajectory
        b (_Trajectory): The second trajectory
        w (int): width of level
        h (int): height of level
        n_trajectory_samples (int, optional): Number of samples to generate uniformly along each trajectory. This is not used. Defaults to 30.

    Returns:
        float: The diversity from these two levels.
    """
    p1 = make_probability_distribution_from_trajectory(a, w, h, True)
    p2 = make_probability_distribution_from_trajectory(b, w, h, True)
    total_value = 0
    for v1, v2 in zip(p1, p2):
        total_value += (v1 - v2) ** 2
    return total_value

def simplified_norm_trajectory_comparison(a: _Trajectory, b: _Trajectory, w: int, h: int, n_trajectory_samples: int = 30) -> float:
    """
        Computes the total euclidean distance between a[i] and b[i] for all i.
        It pads a or b with their ending value to ensure the lengths are the same.

    Args:
        a (_Trajectory): The first trajectory
        b (_Trajectory): The second trajectory
        w (int): width of level
        h (int): height of level
        n_trajectory_samples (int, optional): Number of samples to generate uniformly along each trajectory. This is not used. Defaults to 30.

    Returns:
        float: The diversity from these two levels, between 0 and 1
    """
   # Remove duplicates
    def remove_dups(pos):
        return pos
        new = []
        for p in pos:
            if p not in new: new.append(p)
        return new
    pos1 = remove_dups(a)
    pos2 = remove_dups(b)
    if len(pos1) > len(pos2):
        # Make sure pos1 has the smaller length
        pos1, pos2 = pos2, pos1
    
    # Pad shorted one with the endpoint
    if len(pos1) < len(pos2):
        pos1.extend([pos1[-1]] * (len(pos2) - len(pos1)))
    assert len(pos1) == len(pos2)
    
    def d(a, b):
        # Manhattan distance
        return np.linalg.norm(np.array(a) - np.array(b), ord=1)
    
    all_dists = []
    norm = np.linalg.norm(np.array([w, h]), ord=1)
    for index, (a, b) in enumerate(zip(pos1, pos2)):
        all_dists.append(d(a, b) / (norm))
    
    return (sum(all_dists) / len(all_dists))

def compare_actions_edit_distance(a: List[int], b: List[int], w: int, h: int, n_trajectory_samples: int = 30) -> float:
    """This compares the edit distance between action strings.

    Args:
        a (List[int]): List of actions taken
        b (List[int]): List of actions taken
        w (int): width
        h (int): height
        n_trajectory_samples (int, optional): Not used. Defaults to 30.

    Returns:
        float: The edit distance between two trajectories
    """

    # Use get_char here, as that handles
    # different sized numbers. Very important.
    def get_char(x):
        if x < 10: return str(x)
        Y = x - 10 + 65
        assert 65 <= Y < 65 + 26
        return chr(Y)
    sa = ''.join(map(get_char, a))
    sb = ''.join(map(get_char, b))
    return levenshtein_distance(sa, sb) / max(len(sa), len(sb))


