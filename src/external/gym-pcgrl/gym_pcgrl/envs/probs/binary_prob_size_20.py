import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from games.maze.maze_level import MazeLevel

from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_num_regions, calc_longest_path
from novelty_neat.maze.utils import path_length

"""
Michael: We add in this one to see how PCGRL generates levels that are larger than the stock 20x20 one.
This is a simple copy-paste, just to make it separate.

Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
class BinaryProblemSize20(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 20
        self._height = 20
        self._prob = {"empty": 0.5, "solid":0.5}
        self._border_tile = "solid"

        self._target_path = 20
        self._random_probs = True

        self._rewards = {
            "regions": 5,
            "path-length": 2
        }
    def abc(self):
        print("IN ABC TEST")
        pass
    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["empty", "solid"]

    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile
        intiialization, the names are "empty", "solid"
        target_path (int): the current path length that the episode turn when it reaches
        rewards (dict(string,float)): the weights of each reward change between the new_stats and old_stats
    """
    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        self._target_path = kwargs.get('target_path', self._target_path)
        self._random_probs = kwargs.get('random_probs', self._random_probs)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]

    """
    Resets the problem to the initial state and save the start_stats from the starting map.
    Also, it can be used to change values between different environment resets

    Parameters:
        start_stats (dict(string,any)): the first stats of the map
    """
    def reset(self, start_stats):
        super().reset(start_stats)
        if self._random_probs:
            self._prob["empty"] = self._random.random()
            self._prob["solid"] = 1 - self._prob["empty"]

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "reigons": number of connected empty tiles, "path-length": the longest path across the map
    """
    def get_stats(self, map):
        assert len(map) == 20
        map_locations = get_tile_locations(map, self.get_tile_types())
        np_array_map = np.zeros((self._height, self._width))
        for y in range(len(map)):
            for x in range(len(map[0])):
                np_array_map[y, x] = 0 if map[y][x] == 'empty' else 1
        level = MazeLevel.from_map(np_array_map)
        p_length = path_length(level)
        return {
            "regions": calc_num_regions(map, map_locations, ["empty"]),
            "path-length": p_length # calc_longest_path(map, map_locations, ["empty"])
        }

    """
    Get the current game reward between two stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        float: the current reward due to the change between the old map stats and the new map stats
    """
    def get_reward(self, new_stats, old_stats):
        # If we remove a path, then give a very low reward.
        if new_stats["path-length"] < 0 and old_stats["path-length"] > 0:
            return -100
        rewards = {
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
            "path-length": get_range_reward(new_stats["path-length"], old_stats["path-length"], 20, 80)
        }
        #calculate the total reward
        return rewards["path-length"] * self._rewards['path-length'] + rewards["regions"] * self._rewards['regions']

    """
    Uses the stats to check if the problem ended (episode_over) which means reached
    a satisfying quality based on the stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        boolean: True if the level reached satisfying quality based on the stats and False otherwise
    """
    def get_episode_over(self, new_stats, old_stats):
        return new_stats["path-length"] - self._start_stats["path-length"] >= self._target_path

    """
    Get any debug information need to be printed

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        dict(any,any): is a debug information that can be used to debug what is
        happening in the problem
    """
    def get_debug_info(self, new_stats, old_stats):
        return {
            # "regions": new_stats["regions"],
            "path-length": new_stats["path-length"],
            'test': 1,
            "path-imp": new_stats["path-length"] - self._start_stats["path-length"]
        }

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using the binary graphics
    """
    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/binary/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/binary/solid.png").convert('RGBA')
            }
        return super().render(map)
