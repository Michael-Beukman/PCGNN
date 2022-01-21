import glob
import os
import pickle
from typing import List
import cv2
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage.morphology as morph
from common.utils import mysavefig
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarSolvabilityMetric
from metrics.horn.leniency import LeniencyMetric
from tqdm import tqdm
from scipy.stats import mannwhitneyu
import seaborn as sns
sns.set_theme()
"""
    This file performs the experiment where we look at whether the metrics also return the same difficulty scaling as expected from the ground truth labels.
    
    We use both https://github.com/mwmancuso/Personal-Maze and http://www.glassgiant.com/maze/ as sources of mazes, and this code processes these files and evaluates them.
    
    
    For the former, the code is located at src/external/metrics_maze_generation/Personal-Maze. Look at the generate_all.sh file. The results can be found in src/runs/proper_experiments/v300_metrics/personal_maze.
    
    For glassgiant, we do not publicly release the levels, but these can be generated using the site linked above.
"""




GLASSGIANT_DIFFICULTIES = ['very_easy', 'easy', 'moderate', 'difficult', 'very_difficult']
NAMES = list(map(str, range(1, 11)))


def read_file(name: str) -> MazeLevel:
    """Reads in a file that was generated using https://github.com/mwmancuso/Personal-Maze and converts it to a maze level.

    Args:
        name (str): The filename

    Returns:
        MazeLevel: A level
    """
    with open(name, 'r') as f:
        arr = None
        rows = f.readlines()
        p = []
        for i, r in enumerate(rows):
            t = []
            r = r.strip("\n")
            if r == '': break
            for j, c in enumerate(r):
                t.append(1 if c == '#' else 0)
            p.append(t)
        arr = np.array(p)
        
        # Find start and end - a boundary tile that is not a wall.
        # We make the start/end then just offset from the boundary by one tile
        locs = []
        for i in range(arr.shape[0]):
            if arr[i, 0] == 0:
                locs.append((1, i))
            if arr[i, -1] == 0:
                locs.append((arr.shape[1] - 2, i))
            
        for i in range(arr.shape[1]):
            if arr[0, i] == 0:
                locs.append((i, 1))
            if arr[-1, i] == 0:
                locs.append((i, arr.shape[0] - 2))
        
            
        # Make boundary solid wall
        arr[0, :] = 1
        arr[-1, :] = 1
        arr[:, 0] = 1
        arr[:, -1] = 1
        
        assert len(locs) == 2
        for l in locs:
            assert arr[l[1], l[0]]  == 0
        
        
        connected = morph.label(arr + 1, connectivity=1)
        start, end = locs
        # Assert start and end are connected, and empty.
        assert (connected[start[1], start[0]] == connected[end[1], end[0]] and arr[end[1], end[0]] == 0)
        
        return MazeLevel.from_map(arr, start=start, end=end)


def read_glassgiant_maze(filename: str, index: int, 
                        logo=False,
                        WIDTH: int = 20, HEIGHT: int = 20) -> MazeLevel:
    """Reads in a glassgiant image maze and returns a mazelevel.

    Args:
        filename (str): .png filename
        index (int): An index that will be the filename of the .p file
        logo (bool, optional): If true, there is a logo that takes 40 pixels at the bottom of the screen. Defaults to False.
        WIDTH (int, optional): Width of the maze. Defaults to 20.
        HEIGHT (int, optional): Height of the maze. Defaults to 20.

    Returns:
        MazeLevel: [description]
    """
    SIZE = 15
    
    arr = cv2.imread(filename)[:, :, ::-1]
    where_to_save = os.path.join(*filename.replace("raw", "processed").split(os.sep)[:-1])
    os.makedirs(where_to_save, exist_ok=True)
    arr = arr[:-1 - logo*40, :-1]
    BLACK, GREEN, RED, WHITE = np.array([0, 0, 0]), np.array([0, 204, 0]), np.array([204, 0, 0]), np.array([255, 255, 255])
    uniques = list(map(tuple, np.unique(arr.reshape(-1, arr.shape[2]), axis=0)))
    assert set(uniques) == set(map(tuple, [BLACK, GREEN, RED, WHITE])) or set(uniques) == set(map(tuple, [BLACK, WHITE]))
    a, b, _ = (np.where(arr == BLACK))
    arr = arr[min(a): max(a), min(b): max(b)]
    arr = arr[2:, 2:]
    
    
    def coords(x, y):
        a = x * SIZE + 1
        b = y * SIZE + 1
        
        c = (x+1) * SIZE - 2
        d = (y+1) * SIZE - 2
        return a, b, c, d
    
    def generate_maze(arr):
        new_map = np.zeros((HEIGHT * 2 , WIDTH * 2))
        start, end = None, None
        for x in range(WIDTH):
            for y in range(HEIGHT):
                # So, this loc is always white 
                a, b, c, d = coords(x, y)
                # check to the right if there is a wall
                
                
                center_x = (a + c)//2
                center_y = (b + d)//2
                length_x =  abs(a - c)//2 + 1
                length_y =  abs(b - d)//2 + 1
                
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            for l in range(-1, 2):
                                
                                tx_glass, ty_glass = int(center_x + (i + k / 3) * length_x), int(center_y + (j + l / 3) * length_y)
                                if tx_glass >= 0 and tx_glass < arr.shape[1] and ty_glass >= 0 and ty_glass < arr.shape[0]:
                                    if tuple(arr[ty_glass, tx_glass]) == tuple(BLACK):
                                        assert j != 0 or i != 0
                                        if j != 0 or i != 0:
                                            new_map[2 * y + j, 2 * x + i] = 1 

        M = np.ones((HEIGHT * 2 + 1, WIDTH * 2 + 1))
        M[1:-1, 1:-1] = new_map[:-1, :-1]
        start = (1, 1)
        end = (WIDTH*2 - 1, HEIGHT*2 - 1)
        return M, start, end

    new_map, start, end = generate_maze(arr)
    level = MazeLevel.from_map(new_map, start=start, end=end)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(arr)
    axs[1].imshow(1 - new_map, cmap='gray', vmin=0, vmax=1)
    plt.suptitle(filename)
    mysavefig(os.path.join(where_to_save, f'{index}.png'))
    with open(os.path.join(where_to_save, f'p_{index}.p'), 'wb+') as f:
        pickle.dump(level, f)
    plt.close()



# Difficulty
    
def process():
    """
        Processes all of the text files into mazelevel pickle files.
    """
    for n in NAMES:
        D = f'runs/proper_experiments/v300_metrics/personal_maze/raw/{n}'
        prev = f'{D}/*'
        os.makedirs(D.replace('raw', 'processed'), exist_ok=True)
        files = glob.glob(prev)
        for f in tqdm(files):
            to_save = f.replace('raw', 'processed')
            level = read_file(f)
            with open(to_save + ".p", 'wb+') as f:
                pickle.dump(level, f)


def process_glassgiant():
    for x in GLASSGIANT_DIFFICULTIES:
        files = sorted(glob.glob(f'runs/proper_experiments/v300_metrics/glassgiant_mazes/square/raw/{x}/*.png'))
        for i, f in enumerate(files):
            read_glassgiant_maze(f, i, logo=True)

def evaluate_all(NAMES: List[str], loc: str, root = 'runs/proper_experiments/v300_metrics/personal_maze', do_len=False, max_n=30):
    """Runs through the entire directory, with the names specified in NAMES and evaluates them.
    
        Finally plots a few heatmaps

    Args:
        NAMES (List[str]): Difficulty names, i.e. names of folders in `root`.
        loc (str): The location to save the plots to.
        root (str, optional): Where are the difficulty folders located. Defaults to 'runs/proper_experiments/v300_metrics/personal_maze'.
        do_len (bool, optional): If true, does leniency, otherwise A* difficulty. Defaults to False.
        max_n (int, optional): Maximum number of levels per class. Defaults to 30.
    """
    answers = {}
    levels = {}
    for d in tqdm(NAMES):
        answers[d] = []
        levels[d] = []
        files = natsorted(glob.glob(f'{root}/processed/{d}/*.p'))[:max_n]
        for f in (files):
            with open(f, 'rb') as file:
                level: MazeLevel = pickle.load(file)
                levels[d].append(level)
                W = level.width
                
        # Metrics
        game = MazeGame(MazeLevel(W, W))
        parent = AStarDiversityAndDifficultyMetric(game)
        metric = AStarDifficultyMetric(game, parent)
        if do_len:
            metric = LeniencyMetric(game)
        # Evaluate
        answer = metric.evaluate(levels[d])
        answers[d] = answer
    
    
    # Now plot and evaluate results
    print(metric.name())
    print('=' * 100)
    print('\n'.join(f'{k:<20}: {np.round(np.mean(v), 2):<4} +- {np.round(np.std(v), 2)}' for k, v in answers.items()))
    print('=' * 100)
    
    keys = list(answers.keys())
    vals = list(answers.values())
    
    arr = np.empty((len(keys), len(keys)), dtype=np.float32)
    # Compare difficulty classes to the subsequent ones to see if later difficulties get measured as more difficult by the metrics.
    mask = 1 - np.triu(np.ones_like(arr))
    for i, k in enumerate(keys):
        arr[i, i] = np.round(np.mean(vals[i]), 2)
        for j, k2 in enumerate(keys[i+1:]):
            v1 = vals[i]
            v2 = vals[i + 1 + j]
            
            stat, p = mannwhitneyu(v1, v2, alternative='less' if not do_len else 'greater')
            stat = np.round(stat, 1)
            A = 'YES' if p < 0.05 else 'NO'
            p = np.round(p, 3)
            if j == 0:
                print(f"For {k:<20} vs {k2:<20}, U = {stat:<20}. P = {p:<8} {A:<20}")
            else:
                print(f"    {'':<20} vs {k2:<20}, U = {stat:<20}. P = {p:<8} {A:<20}")
            arr[i, i + 1 + j] = p
    
    pretty = list(map(lambda s: s.replace("_", " ").title(), NAMES))
    df = pd.DataFrame(arr, index=pretty, columns=pretty)
    sns.heatmap(df, annot=arr, mask=mask, cbar=False, vmin=0, vmax=1)
    os.makedirs(loc, exist_ok=True)
    mysavefig(os.path.join(loc, f'heatmap_mann_whitney{"_len" if do_len else ""}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    # Do this if necessary
    # process()
    # process_glassgiant()
    if 1:
        for do_len in [False, True]:
            evaluate_all(NAMES, loc='results/v300/metrics/personal_maze_diff', do_len=do_len, max_n=1000)
    
    if 1:
        for do_len in [False, True]:
            evaluate_all(['very_easy', 'easy', 'moderate', 'difficult', 'very_difficult'], root='runs/proper_experiments/v300_metrics/glassgiant_mazes/square', loc='results/v300/metrics/glassgiant_diff', do_len=do_len)
            plt.close()
    