import datetime
import os
from typing import List, Tuple, Union
import subprocess
from timeit import default_timer as timer
from matplotlib import pyplot as plt

import numpy as np
from common.utils import get_date
from games.mario.mario_game import MarioGame

from games.mario.mario_level import MarioLevel
"""This file contains some utility code to run the java RL agents.
"""
PATH = "external/Mario-AI-Framework/"
J = os.path.join


def run_java_task(args: List[str]) -> str:
    all_args = ["java", "--enable-preview", "-jar", J(PATH, "Mario-AI-Framework.jar")] + args

    ans = subprocess.run(all_args, capture_output=True)
    return ans.stdout.decode("utf-8")


def write_level_to_file(level: MarioLevel):
    # filename = f'results/mario/levels/{get_date()}-{np.random.randint(1000, 9999)}-{os.getpid()}'
    filename = f'/tmp/{get_date()}-{np.random.randint(1000, 9999)}-{os.getpid()}'
    level.to_file_for_java(filename)
    return filename


def java_astar_number_of_things_in_open_set(level: MarioLevel, time_per_episode=20, filename: str=None) -> Tuple[bool, List[Tuple[int, int]], List[int], float]:
    """Returns if the level is solvable and how many total states were visited by A*.

    Args:
        level (MarioLevel): [description]
        time_per_episode (int, optional): [description]. Defaults to 20.

    Returns:
        Tuple[bool, List[int, int], List[int], float]: solvable, trajectory, actions, num_states (normalised by dividing by width * height * time_per_episode * 1000 / 30)
    """
    if filename is None:
        filename = write_level_to_file(level)
    args = ["Astar_Solvability", filename, str(time_per_episode), str(1), 'true']
    string = run_java_task(args)
    lines = string.split("\n")
    result_line = [k for k in lines if 'sum' in k.lower()]
    solv_line = [l for l in lines if 'Result' in l]

    # trajectories
    traj_line = [l for l in lines if 'Trajectories' in l]
    action_trajectories_line = [l for l in lines if 'Actions' in l]
    
    expanded_line = [l for l in lines if 'NumberOfStatesExpanded' in l]



    if len(result_line) == 0 or len(solv_line) == 0 or len(traj_line) == 0 or len(action_trajectories_line) == 0 or len(expanded_line) == 0:
        raise Exception("Java didn't print out result properly: " + string + "args = " + ' '.join(args))
    else:

        # more trajectories

        vals = [s.strip() for s in traj_line[0].split(":")[1].split(" ")]
        vals = [s for s in vals if s != '']
        trajectories = [tuple(map(lambda x: int(float(x) / 16), s.split(','))) for s in vals]

        # action trajs

        vals = [s.strip() for s in action_trajectories_line[0].split(":")[1].split(" ")]
        vals = [s for s in vals if s != '']

        action_trajectories = [int(s) for s in vals]

        sum_number_states_considered = (float(result_line[0].split(":")[-1]))
        
        # now, we normalise this as follows:
        # each two steps count as one, as we had a look ahead of 2.
        # Thus, for each step the max to do is width * height * time
        # we also subtract the actual path length.
        
        norms = len(trajectories) / 2 * (16) * (16) * 10 - len(action_trajectories)
        sum_number_states_considered = (sum_number_states_considered - len(action_trajectories)) / norms

        expandeds = [int(s.strip()) for s in expanded_line[0].split(":")[1].split(" ") if s.strip() != '']

        solv = 'WIN' in solv_line[0]
        return solv, trajectories, action_trajectories, sum_number_states_considered

def java_solvability(level: MarioLevel, time_per_episode=20, verbose=False, return_trajectories=False) -> Union[bool, Tuple[bool, List[Tuple[float, float]]]]:
    """Returns a boolean indicating if this level is solvable.

    Args:
        level (MarioLevel): The level
        time_per_episode (int, optional): How many seconds per episodes. Defaults to 500.
        verbose (bool, optional): Should this print many info. Defaults to False.
        return_trajectories (bool, optional). If this is true, then we are by default verbose and we return trajectories
    
    Returns:
        Union[bool,                             :Is this solvable
              Tuple[bool, List[Tuple[float, float]]]   : solvable, trajectory if return_trajectories = True
        ]
    """
    filename = write_level_to_file(level)
    verbose = verbose or return_trajectories
    args = ["Astar_Solvability", filename, str(time_per_episode), str(1), str(verbose).lower()]
    s = timer()
    string = run_java_task(args)
    e = timer()

    lines = string.split("\n")
    result_line = [l for l in lines if 'Result' in l]
    if len(result_line) == 0:
        raise Exception("Java didn't print out result properly: " + string + "args = " + ' '.join(args))
    if return_trajectories:
        traj_line = [l for l in lines if 'Trajectories' in l]
        if len(traj_line) == 0:
            raise Exception("Java didn't print out trajectory properly: " + string + "args = " + ' '.join(args))
        vals = [s.strip() for s in traj_line[0].split(":")[1].split(" ")]
        vals = [s for s in vals if s != '']
        vals = [ tuple(map(float, s.split(','))) for s in vals]
        return 'WIN' in result_line[0], vals
    return 'WIN' in result_line[0]


def java_rl_diversity(level: MarioLevel, number_of_episodes = 2000, time_per_episode = 500, verbose=False) -> Tuple[
    bool, List[int], List[int], List[Tuple[int, int]]
]:
    """Runs the RL diversity metric

    Args:
        level (MarioLevel): Level
        number_of_episodes (int, optional): How many episodes to run. Defaults to 2000.
        time_per_episode (int, optional): How long per episode does Mario have available. Defaults to 500.
        verbose (bool, optional): . Defaults to False.

    Returns:
        Tuple[ bool, List[int], List[int], List[Tuple[int, int]] ]: 
            - has solved the level in evaluation
            - list of actions
            - list of evaluation
            - (x, y) positions
    """
    filename = write_level_to_file(level)
    args = ["RL_Diversity", filename, str(time_per_episode), str(number_of_episodes), str(verbose).lower()]
    string = run_java_task(args)
    lines = string.split("\n")
    result_line = [l for l in lines if 'Result' in l]
    if len(result_line) < 4:
        raise Exception("Java didn't print out result properly: " + string + "args = " + ' '.join(args))

    is_win = 'WIN' in result_line[0]
    actions = list(map(int, filter(lambda x: x != '', result_line[1].split(": ")[1].split(" "))))
    states = list(map(int, filter(lambda x: x != '', result_line[2].split(": ")[1].split(" "))))
    positions = [tuple(map(int, r.split(','))) for r in filter(lambda x: x != '', result_line[3].split(': ')[1].split(' '))]
    return is_win, actions, states, positions


def java_rl_difficulty(level: MarioLevel, number_of_episodes = 2000, time_per_episode = 200, verbose=False) -> float:
    filename = write_level_to_file(level)
    args = ["RL_Difficulty", filename, str(time_per_episode), str(number_of_episodes), str(verbose).lower()]
    string = run_java_task(args)
    lines = string.split("\n")
    result_line = [l for l in lines if 'Result' in l]
    if len(result_line) == 0:
        raise Exception("Java didn't print out result properly: " + string + "args = " + ' '.join(args))

    answer = float(result_line[0].split(": ")[1].strip())
    return answer

def _get_mario_test_levels():
    level_solv = MarioLevel()
    level_not_solv = MarioLevel()
    level_not_solv.map[-1, 10:12] = 0

    for index, x in enumerate([1, 2, 3, 4, 5]):
         level_not_solv.map[-1, 14 + index] = 0
         level_not_solv.map[-index - 1, 14 + index] = 1

    for index, x in enumerate([1, 2, 3, 4, 5]):
         level_not_solv.map[-1, 14 + 5 + index] = 0
         if index > 0:
            level_not_solv.map[-5, 14 + 5 + index] = 1

    level_2 = MarioLevel()
    level_2.map[-1, 10:30] = 0
    level_not_solv = level_2

    return level_solv, level_not_solv


if __name__ == '__main__':
    def test_solv_time():
        from matplotlib import pyplot as plt
        from games.mario.assets.engine import MarioAstarAgent
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

        def get_ans_python(level):
            state = State()
            state.stringInitialize(level.string_representation_of_level().split("\n"))

            aStarAgent = MarioAstarAgent()

            sol,solState,iters = aStarAgent.getSolution(state, 1, 10000)
            if solState.checkWin():
                return 1
            sol,solState,iters = aStarAgent.getSolution(state, 0, 10000)
            if solState.checkWin():
                return 1
            return 0
            
        def get_ans_java(level):
            return java_solvability(level)
        level, _ = _get_mario_test_levels()
        N = 10
        time_java = 0
        time_python = 0
        for i in range(N):
            s = timer()
            get_ans_java(level)
            e = timer()
            time_java += e - s
            
            s = timer()
            get_ans_python(level)
            e = timer()
            time_python += e - s
        print(f"PYTHON = {time_python}s, Java = {time_java}s")
        
    def test_solv():
        solvable, not_solv = _get_mario_test_levels()
        s = timer()
        print("Solvable result Level 1 = ", java_solvability(solvable))
        e = timer()
        print("TIME = ", e - s)
        
        from metrics.solvability import SolvabilityMetric
        m = SolvabilityMetric(MarioGame(MarioLevel()))
        s = timer()
        print("Solvable result Level 1 = ", m.evaluate_mario([solvable]))
        e = timer()
        print("TIME = ", e - s)
        print("Solvable result Level 2 = ", java_solvability(not_solv))
    
    def test_diversity():
        solvable, not_solv = _get_mario_test_levels()
        solv_results = java_rl_diversity(solvable)
        print(solv_results)
        solvable.show()
        # for x, y in solv_results[-1]:
        abc = np.array(solv_results[-1])
        X, Y = zip(*abc)
        plt.plot(np.array(X) + 3 * 16, Y, color='black')
        plt.show()
        not_solv_results = java_rl_diversity(not_solv)
        print(not_solv_results)
        plt.figure()
        not_solv.show()
        abc = np.array(not_solv_results[-1])
        X, Y = zip(*abc)
        plt.plot(np.array(X) + 3 * 16, Y, color='black')
        plt.show()
        not_solv_results = java_rl_diversity(not_solv)

    def test_difficulty():
        solvable, not_solv = _get_mario_test_levels()
        solv_results = java_rl_difficulty(solvable)
        print(solv_results)
        not_solv_results = java_rl_difficulty(not_solv)
        print(not_solv_results)

    def test_trajectory():
        solvable, not_solv = _get_mario_test_levels()
        s = timer()
        solvs1, trajs1 = java_solvability(solvable, return_trajectories=True)
        solvs2, trajs2 = java_solvability(not_solv, return_trajectories=True)
        fig, axs = plt.subplots(2, 1, figsize=(40, 16))
        axs[0].imshow(solvable.show(False))
        axs[0].plot(*(np.array(list(zip(*trajs1))) + np.reshape([3 * 16, -16], (2, 1))), color='black', linewidth=2)
        axs[0].set_title(f"Solv = {solvs1}")
        
        axs[1].imshow(not_solv.show(False))
        axs[1].plot(*(np.array(list(zip(*trajs2))) + np.reshape([3 * 16, -16], (2, 1))), color='black', linewidth=2)
        axs[1].set_title(f"Solv = {solvs2}")
        plt.show()

    def test_a_star():
        solvable, not_solvable = _get_mario_test_levels()
        a, b, c, d = java_astar_number_of_things_in_open_set(solvable)
        print('ConsideredSolv=', d)

        a, b, c, d = java_astar_number_of_things_in_open_set(not_solvable)
        print('ConsideredNotSolv=', d)
    # choose one
    test_a_star(); exit()
    # test_trajectory(); exit()
    # test_solv_time(); exit()
    # test_solv(); exit()
    # test_difficulty(); exit()
