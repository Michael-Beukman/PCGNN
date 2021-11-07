import pickle
import random
from typing import List
from matplotlib import pyplot as plt

import numpy as np
from games.game import Game
from games.gym_wrapper import GymMarioWrapper, GymMazeWrapper
from games.level import Level
from games.mario.java_runner import java_rl_difficulty
from games.mario.mario_game import MarioGame
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.horn.leniency import LeniencyMetric
from metrics.metric import Metric
from metrics.rl.tabular.rl_agent_metric import _FixedLevelGenerator
from metrics.rl.tabular.tabular_rl_agent import TabularRLAgent


class RLDifficultyMetric(Metric):
    def __init__(self, game: Game, N: int = 200) -> None:
        super().__init__(game)
        self.N = N
    def _evaluate_mario(self, levels: List[Level]) -> List[float]:
        ans = []
        for l in levels:
            ans.append(java_rl_difficulty(l, number_of_episodes = self.N))
        return ans
    def evaluate(self, levels: List[Level]) -> List[float]:
        if (isinstance(self.game, MarioGame)):
            return self._evaluate_mario(levels)
        number_of_steps_for_convergence = []
        path_lengths = []

        alpha = 1
        w = levels[0].map.shape[0]
        if w <= 20:
            timestep_cap = 50
        elif w <= 40:
            timestep_cap = 100
        elif w <= 105:
            timestep_cap = 500
        else:
            timestep_cap = levels[0].map.size / 2 # 1000
        solved = lambda eval_rewards: all([e[-1] > 10 for e in eval_rewards])
        num_states = lambda env, level: level.map.size
        if (isinstance(self.game, MarioGame)):
            print("USING BAD METRIC FOR MARIO")
            alpha = 0.5
            timestep_cap = 500
            num_states = lambda env, level: env.observation_space.n
            # solved = lambda eval_rewards: all([e[-1] > -timestep_cap for e in eval_rewards])

        if 1 or (isinstance(self.game, MazeGame)):
            for level in levels:
                gen = _FixedLevelGenerator(level)
                # env = GymMazeWrapper(level_generator=gen, timestep_cap=100, number_of_level_repeats=0, init_level=level)
                if (isinstance(self.game, MazeGame)):
                    env = GymMazeWrapper(level_generator=gen, timestep_cap=timestep_cap, number_of_level_repeats=0, init_level=level)
                else:
                    env = GymMarioWrapper(level_generator=gen, timestep_cap=timestep_cap, number_of_level_repeats=0, init_level=level)
                # agent = TabularRLAgent(level.map.size, env.action_space.n)
                agent = TabularRLAgent(num_states(env, level), env.action_space.n, alpha=alpha)
                prev_r = -100000000
                # If we aren't solvable at all, return 1.
                if level.map[0, 0] == level.tile_types_reversed['filled'] or level.map[-1, -1] == level.tile_types_reversed['filled']:
                    i = self.N
                else:
                # This ranges from [1, N]
                    for i in range(1, self.N + 1):
                        alpha, epsilon = agent.alpha, agent.epsilon
                        agent.train(env, episodes=5)
                        # plt.imshow(np.max(agent.table, axis=1).reshape(14, 14))
                        # plt.show()
                        evals = agent.eval_difficulty(env, 5)
                        # evals = []
                        agent.alpha = alpha
                        agent.epsilon = epsilon
                        
                        # has solved = have all runs resulted in positive = goal score? 
                        # has_solved = all([e[-1] > 10 for e in evals])
                        has_solved = solved(evals)
                        length_per_ep = max([len(e) for e in evals])
                        rewards_per_ep = [np.sum(e) for e in evals]
                        all_rs = np.mean(rewards_per_ep)

                        # If we have converged and solved the env, then return.
                        if np.abs(all_rs - prev_r) < 1 and has_solved:
                            break

                        prev_r = all_rs
                assert i > 0
                # i - 1 ranges from 0 to N - 1
                diff = (i - 1) / (self.N - 1)
                number_of_steps_for_convergence.append(diff)
                # path_lengths.append(length_per_ep)
            return number_of_steps_for_convergence
        else:
            pass
        