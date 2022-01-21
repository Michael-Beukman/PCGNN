from collections import defaultdict
from queue import PriorityQueue
from typing import List, Tuple, Union
from matplotlib import pyplot as plt
import numpy as np

from games.maze.maze_level import MazeLevel

_Point = Tuple[int, int]

def a_star(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], blocked_tile: int) -> Tuple[Union[List[Tuple[int, int]], None], int]:
    # source: wikipedia https://en.wikipedia.org/wiki/A*_search_algorithm
    def get_path(curr):
        path = [curr]
        while 1:
            curr = tuple(came_from[curr[1]][curr[0]])
            if (curr[0] == -1): break
            path.append(curr)
    
        return list(reversed(path))

    def calc_dist(a, b):
        return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

    def h(pos):
        ans = calc_dist(pos, goal)
        return ans
        return np.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)


    def neighbours(pos: _Point) -> List[_Point]:
        ans = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == j == 0 or abs(i) == abs(j) == 1: continue
                nx = pos[0] + i
                ny = pos[1] + j

                if nx < 0 or ny < 0 or nx >= grid.shape[1] or ny >= grid.shape[0]:
                    # invalid bounds
                    continue
                if grid[ny][nx] == blocked_tile:
                    # wall
                    continue
                
                ans.append((nx, ny))
        return ans
    open_set = PriorityQueue()
    open_set.put((h(start), 0, start))
    is_in_open = {start}
    came_from = np.zeros((grid.shape[0], grid.shape[1], 2), dtype=np.int32) - 1;

    g_score = defaultdict(lambda: np.inf)
    g_score[start] = 0
    
    f_score = defaultdict(lambda: np.inf)
    f_score[start] = h(start)
    number_of_things_considered = 0
    all_sets = {(0, start)}
    visited = set()
    K = 0
    
    open = {start}

    while not open_set.empty():
        K+=1
        dist, _, pos = open_set.get()
        number_of_things_considered += 1
        if pos in visited: continue
        visited.add(pos)
        is_in_open.remove(pos)
        if pos == goal: 
            return get_path(pos), visited, len(visited)
        ns = neighbours(pos)
        for n in ns:
            if n in visited: continue
            tentative_score = g_score[pos] + calc_dist(pos, n)
            if tentative_score < g_score[n]:
                came_from[n[1]][n[0]] = pos
                g_score[n] = tentative_score
                f_score[n] = g_score[n] + h(n)
                all_sets.add((K, n))

                if n not in open:
                    open.add(n)
                    open_set.put((f_score[n], 100-K, n))
                    is_in_open.add(n)
    
    return None, visited, len(visited)
    # return None, all_sets, number_of_things_considered + open_set.qsize()
    
def do_astar_from_level(level: MazeLevel) -> Tuple[Union[List[Tuple[int, int]], None], int]:
    height, width = level.map.shape
    goal = level.end # (width - 1, height - 1)
    if not hasattr(level, 'tile_types_reversed'):
        level.tile_types_reversed = {v: k for k, v in level.tile_types.items()}
    # cannot start on a filled up tile.
    if level.map[level.start[1], level.start[0]] == level.tile_types_reversed['filled']:
        return None, set(), (level.map == level.tile_types_reversed['empty']).sum()
    path = a_star(level.map, level.start , goal, level.tile_types_reversed['filled'])
    return path

if __name__ == '__main__':
    solvable_map = np.array([
        [0, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 0]
    ])
    

    empty = np.array([
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

    test = np.array([
        [0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0]
    ])

    level = np.zeros((20, 20))

    i = np.arange(len(level) - 1)
    for k in range(20):
        modder = k % 2
        divver = k // 2 % 2
        if modder == 1 and k != 19:
            level[k, i-(1 if divver == 1 else -1)] = 1

    def test_single(level, name):
        L = MazeLevel.from_map(level)
        path, sets, count = do_astar_from_level(L)
        level = 1 - level
        for (x, y) in sets:
            level[y][x] = 3
        X, Y = zip(*path)
        level[Y, X] = 3
        plt.imshow(level)
        plt.plot(X, Y)
        plt.colorbar()
        
        num_passable_tiles = max(1, (L.map == L.tile_types_reversed['empty']).sum() - len(path))
        plt.title(f"Number = {(count - len(path))/num_passable_tiles} vs {(count)/max(1, (L.map == L.tile_types_reversed['empty']).sum())}")
        plt.show()
    
    test_single(level, 'a_star_long_path')
    empty_level = np.zeros((20, 20))
    test_single(empty_level, 'a_star_empty')

    test = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    test_single(test, 'test')
    