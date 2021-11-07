from typing import Dict, Tuple, Union

import numpy as np
from common.types import TileMap


class Level:
    """A level is a single map that consist of a 2d grid of tiles.
    """

    def __init__(self, width: int, height: int, tile_types: Dict[int, str] = {0: 'empty'}, map: Union[None, TileMap] = None):
        self.width = width
        self.height = height
        if map is None:
            map: TileMap = np.zeros((height, width))
        self.map = map
        self.tile_types = tile_types
        self.num_tiles = len(tile_types)
        self.tile_types_reversed = {v: k for k, v in tile_types.items()}

    def show(self) -> None:
        """
            imshows this level.
        """
        raise NotImplementedError()
    
    @staticmethod
    def from_map(map: np.ndarray) -> "Level":
        """Creates a level from an array representing the map.

        Returns:
            Level: 
        """
        raise NotImplementedError()
    def to_file(self, filename: str):
        """Writes this level to a file

        Args:
            filename (str): 
        """
        raise NotImplementedError()