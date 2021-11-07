import numpy as np
from enum import Enum, auto
TileMap = np.ndarray

class Verbosity(Enum):
    NONE     = auto()
    PROGRESS = auto()
    DETAILED = auto()

