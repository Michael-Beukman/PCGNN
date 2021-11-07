from matplotlib import pyplot as plt
import numpy as np
from games.level import Level


class MazeLevel(Level):
    """A simple level consisting of only empty and filled blocks
    """
    def __init__(self, width=14, height=14):
        super().__init__(width, height, tile_types={0: 'empty', 1: 'filled'})
    
    @staticmethod
    def random(width=14, height=14) -> "MazeLevel":
        level = MazeLevel(width, height)
        level.map = (np.random.rand(*level.map.shape) > 0.5).astype(np.int32)
        return level

    
    @staticmethod
    def from_map(map: np.ndarray) -> "MazeLevel":
        """Creates a level from an array representing the map.

        Returns:
            MazeLevel: 
        """
        level = MazeLevel(map.shape[1], map.shape[0])
        level.map = map
        return level
    
    def show(self, do_show: bool = True) -> None:
        if do_show:
            plt.imshow(1 - self.map, cmap='gray', vmin=0, vmax=1)
        return self.map
    
    def to_file(self, filename: str):
        with open(filename, 'w+') as f:
            f.write(self.str())
    
    def str(self) -> str:
        ans = ""
        for row in self.map:
            for c in row:
                ans += '#' if c == 1 else '.'
            ans += '\n'
        return ans