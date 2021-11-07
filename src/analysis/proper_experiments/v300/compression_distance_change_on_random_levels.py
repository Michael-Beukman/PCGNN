from matplotlib import pyplot as plt
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.horn.compression_distance import CompressionDistanceMetric
import seaborn as sns

def main():
    """
    This wants to generate a few mazes of different sizes and show two things:
        - For each size, CD is pretty flat and the histogram is not really spread out
        - CD increases as the size does. This does it for random levels.
    """

    def get_levels(w, N=100):
        return [MazeLevel.random(w, w) for _ in range(N)]
    
    all_levels = []
    cols = ['red', 'green', 'blue', 'orange']
    for i, w in enumerate([14, 30, 80, 200]):
        print(w)
        game = MazeGame(MazeLevel(w, w))
        cd_metric = CompressionDistanceMetric(game)
        all_levels.append(get_levels(w))
        results = cd_metric.evaluate(all_levels[-1])
        sns.histplot(results, label=f'W={w}', color=cols[i])
    # plt.title('W = {w}')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()