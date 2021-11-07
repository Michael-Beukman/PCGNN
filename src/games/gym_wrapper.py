from typing import Generator, final
import gym
from gym import spaces
from matplotlib import pyplot as plt
import numpy as np
from baselines.ga.genetic_algorithm_pcg import GeneticAlgorithmIndividualMaze, GeneticAlgorithmPCG
from games.game import Game

from games.level import Level
from games.mario.mario_game import MarioGame
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel

# LevelGenerator = Generator[Level, None, None]

class LevelGenerator:
    def __init__(self):
        pass
    def get_level(self) -> Level:
        pass

    @final
    def __call__(self) -> Level:
        return self.get_level()

class GymWrapper(gym.Env):
    """A class that wraps a game, and provides the interface that a gym package expects.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, level_generator: LevelGenerator, game: Game, timestep_cap=float('inf'), 
            number_of_level_repeats:int=5, is_tabular: bool = False):
        super().__init__()
        self.game = game
        self.level_generator = level_generator
        self.timestep_cap = timestep_cap
        self.steps = 0
        self.count = 0
        self.number_of_level_repeats = number_of_level_repeats
        self.is_tabular = is_tabular

    def step(self, action):
        # Execute one time step within the environment
        self.steps += 1
        done, reward = self.game.step(action)
        if self.steps >= self.timestep_cap: 
            done = True
        return self.get_obs(), reward, done, {}

    def reset(self):
        self.steps = 0
        if self.count >= self.number_of_level_repeats:
            level = self.level_generator()
            self.count = 0
        else:
            level = self.game.level
            self.count += 1
        self.game.reset(level)
        return self.get_obs()

        
    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            return self.get_obs()
        raise NotImplementedError("Not implementing rendering yet")

    def get_obs(self) -> np.ndarray:
        ans = self.game.level.map.copy().astype(np.int32)
        x, y = self.game.current_pos
        if self.is_tabular:
            return y * self.game.level.width + x
        ans[y, x] = self.game.level.num_tiles # Player
        return ans.flatten()

class GymMazeWrapper(GymWrapper):
    """Wrapper for the Maze Environment
    """
    def __init__(self, level_generator: LevelGenerator, timestep_cap=float('inf'), number_of_level_repeats:int=5, init_level: Level = None):
        self.level_generator = level_generator
        if init_level is None:
            init_level = self.level_generator() 
        game = MazeGame(init_level)
        super().__init__(self.level_generator, game=game, timestep_cap=timestep_cap, number_of_level_repeats=number_of_level_repeats, is_tabular=True)
        self.action_space = spaces.Discrete(4) # 4 directions
        self.observation_space = spaces.Discrete(self.game.level.map.size)


class GymMarioWrapper(GymWrapper):
    """Wrapper for the Maze Environment
    """
    def __init__(self, level_generator: LevelGenerator, timestep_cap=float('inf'), number_of_level_repeats:int=5, init_level: Level = None):
        self.level_generator = level_generator
        if init_level is None:
            init_level = self.level_generator() 
        game = MarioGame(init_level, do_enemies=True)
        super().__init__(self.level_generator, game=game, timestep_cap=timestep_cap, number_of_level_repeats=number_of_level_repeats, is_tabular=True)
        self.action_space = spaces.Discrete(9) # 8 directions and current pos
        # self.observation_space = spaces.Discrete(self.game.level.map.size)
        self.observation_space = spaces.Discrete(self.game.mario_state.width * self.game.mario_state.height)

    def get_obs(self) -> np.ndarray:
        ans = self.game.level.map.copy().astype(np.int32)
        x, y = self.game.current_pos
        if self.is_tabular:
            return y * self.game.mario_state.width + x
        ans[y, x] = self.game.level.num_tiles # Player
        return ans.flatten()

if __name__ == '__main__':
    print("Hello")
    wrapper = GymMazeWrapper()
    state = wrapper.reset()
    for i in range(10):
        print("S = ",state)
        state, reward, done, _ = wrapper.step(1)

        