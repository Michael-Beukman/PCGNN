from common.methods.pcg_method import PCGMethod
from games.game import Game
from games.level import Level
import numpy as np

class RandomBaseline(PCGMethod):
    # This is a simple random baseline
    def __init__(self, game: Game, init_level: Level) -> None:
        super().__init__(game, init_level)
    
    
    def generate_level(self) -> Level:
        low = 0
        high = len(self.init_level.tile_types)
        
        new_map = np.random.randint(low, high, size=self.init_level.map.shape)
        return self.init_level.from_map(new_map)
    
    
if __name__ == '__main__':
    from games.mario.mario_game import MarioGame, MarioLevel
    from games.maze.maze_game import MazeGame
    from games.maze.maze_level import MazeLevel
    import matplotlib.pyplot as plt
    L = MarioLevel()
    baseline = RandomBaseline(MarioGame(L), L)
    
    L = MazeLevel()
    baseline = RandomBaseline(MazeGame(L), L)
    
    plt.imshow(baseline.generate_level().show())
    plt.show()
    plt.imshow(baseline.generate_level().show())
    plt.show()