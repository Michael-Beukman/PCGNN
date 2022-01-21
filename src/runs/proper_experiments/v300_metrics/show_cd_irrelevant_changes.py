import os
from matplotlib import pyplot as plt
import numpy as np
from common.utils import mysavefig
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.a_star.a_star_metrics import AStarDiversityAndDifficultyMetric, AStarEditDistanceDiversityMetric
from metrics.horn.compression_distance import CompressionDistanceMetric
import seaborn as sns
sns.set_theme()
def main():
    """
        This makes a few levels of size X, with one path and a solid block of wall next to it.
        Then, it randomises the rest of the level. The idea is to show that A* measures all the levels to be identical while CD does not.
    """
    W = 30
    
    # How many
    N = 200
    DIR = f'results/v300/metrics/'
    os.makedirs(DIR, exist_ok=True)
    def generate_level(w: int, h: int) -> MazeLevel:
        arr = np.ones((h, w))
        arr[0, :]  = 0
        arr[:, -1] = 0
        arr[2:, :-2] = np.random.rand(h-2, w-2) > 0.5
        return MazeLevel.from_map(arr)
    np.random.seed(42)
    # Get the levels
    levels = [generate_level(W, W) for _ in range(N)]
    
    # Plot some levels
    fig, axs = plt.subplots(2, 2)
    for l, ax in zip(levels, axs.ravel()):
        ax.spines['bottom'].set_color('0')
        ax.spines['top'].set_color('0')
        ax.spines['right'].set_color('0')
        ax.spines['left'].set_color('0')
        ax.tick_params(axis='x', colors='0.7', which='both')
        ax.grid(which='both', alpha=0)
        ax.yaxis.label.set_color('0.9')
        ax.xaxis.label.set_color('0.9')
        ax.imshow(1 - l.map, cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.close()
    
    # Now evaluate the levels
    game = MazeGame(MazeLevel(W, W))
    cd = CompressionDistanceMetric(game)
    parent = AStarDiversityAndDifficultyMetric(game)
    astarmetric = AStarEditDistanceDiversityMetric(game, parent)
    answer = cd.evaluate(levels)
    print("CD Done")
    astar = astarmetric.evaluate(levels)
    print("A* Done")
    
    palette = dict(zip(['CD', 'A*'], sns.color_palette(n_colors=2)))
    dic = {
        'Metric Values': answer + astar,
        'Metric': ['CD']  * len(answer) + ['A*'] * len(astar)
    }
    # And plot
    sns.histplot(dic, x='Metric Values', hue='Metric', palette=palette)
    plt.title("Values of the A* diversity metric and Compression Distance\nfor levels that only differ visually.")
    mysavefig(os.path.join(DIR, 'cd_visual_changes.png'), bbox_inches='tight', pad_inches=0.1)
    
    

if __name__ == '__main__':
    main()