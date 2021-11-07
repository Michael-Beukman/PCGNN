from typing import List, Tuple, Union
from games.maze.maze_level import MazeLevel
import collections
import numpy as np

def shortest_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], blocked_tile: int) -> Union[List[Tuple[int, int]], None]:
    """
    Performs breadth first search on the grid to travel from the start to the goal. If a path does not exist, we return None.
    Otherwise, this function returns a list of coordinates representing the path, including the start and end nodes.

    Some ideas and code from here: https://stackoverflow.com/a/47902476
    Args:
        grid (np.ndarray): The maze to find the path for
        start (Tuple[int, int]): Where to start
        goal (Tuple[int, int]): Where to end
        blocked_tile (int): This tile counts as a 'wall', and will block a path.

    Returns:
        Union[List[Tuple[int, int]], None]: Either None if no path exists or a list of (x, y) coordinates if one does.
    """
    height, width = grid.shape
    queue = collections.deque([[start]])
    seen = set([start])
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if x == goal[0] and y == goal[1]:
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < width and 0 <= y2 < height and grid[y2][x2] != blocked_tile and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))
    return None

def path_length(level: MazeLevel, first_tile: bool = True) -> int:
    """Calculates the shortest path length between coordinate 0, 0 and w-1, h-1.
        Returns -1 if no path exists

    Args:
        level (MazeLevel): The maze level.
        first_tile (bool): If this is true, normal. Otherwise, if the first tile is filled up, no change is made.

    Returns:
        int: The length of the path. -1 if no path exists.


    """
    height, width = level.map.shape
    goal = (width - 1, height - 1)
    if not hasattr(level, 'tile_types_reversed'):
        level.tile_types_reversed = {v: k for k, v in level.tile_types.items()}
    # cannot start on a filled up tile.
    if first_tile and level.map[0, 0] == level.tile_types_reversed['filled']:
        return -1
    path = shortest_path(level.map, (0, 0), goal, level.tile_types_reversed['filled'])
    if path is None:
        return -1
    return len(path)

def get_path_trajectory(level: MazeLevel) -> List[Tuple[int, int]]:
    height, width = level.map.shape
    goal = (width - 1, height - 1)
    if not hasattr(level, 'tile_types_reversed'):
        level.tile_types_reversed = {v: k for k, v in level.tile_types.items()}
    # cannot start on a filled up tile.
    if level.map[0, 0] == level.tile_types_reversed['filled']:
        return []
    path = shortest_path(level.map, (0, 0), goal, level.tile_types_reversed['filled'])
    if path is None:
        return []
    return path

if __name__ == '__main__':
    from timeit import default_timer as tmr
    import skimage.morphology as morph
    def test_performance():
        bfs = 0
        mor = 0
        for i in range(500):
            level = np.random.rand(30, 30) > 0.5
            s = tmr()
            pl = path_length(MazeLevel.from_map(level))
            e = tmr()
            bfs += e - s
            s = tmr()
            pl = morph.label(level)
            e = tmr()
            mor += e - s
        print(f"BFS = {bfs}, Morph = {mor}")
    test_performance()
    solvable_map = np.array([
        [0, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 0]
    ])
    test_map = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    unsolvable_map = np.array([
        [0, 0, 1, 1],
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 0]
    ])

    assert path_length(MazeLevel.from_map(solvable_map)) == 7
    assert path_length(MazeLevel.from_map(test_map)) == 7
    assert path_length(MazeLevel.from_map(unsolvable_map)) == -1
    height, width = solvable_map.shape
    print(shortest_path(MazeLevel.from_map(solvable_map).map, (0, 0), (width-1, height-1), 1))
    pass
