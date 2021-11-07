from typing import Tuple, List
from games.mario.assets.engine import MarioAstarAgent
from games.mario.mario_level import MarioLevel
from gym_pcgrl.envs.probs.smb.engine import State
def _get_mario_path(best_node):
    path = []
    curr = best_node
    while curr:
        path.append(curr.state.player)
        curr = curr.parent
    return list(reversed(path))

def get_path_trajectory_mario(level: MarioLevel) -> List[Tuple[int, int]]:
    """
    Returns a trajectory for the Mario level.
    """
    
    string = level.string_representation_of_level()
    
    state = State()
    state.stringInitialize(string.split("\n"))

    aStarAgent = MarioAstarAgent()

    sol, solState, iters = aStarAgent.getSolution(state, 1, 10000)
    path = _get_mario_path(solState)
    T = [(t['x'], t['y']) for t in path]
    return T