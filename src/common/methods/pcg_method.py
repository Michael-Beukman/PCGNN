from typing import Any, Dict, List
from common.types import Verbosity
from games.game import Game
from games.level import Level
from experiments.logger import Logger

class PCGMethod:
    """A method that can generate levels given a game and initial levels

    """
    def __init__(self, game: Game, init_level: Level) -> None:
        self.game = game
        self.init_level = init_level
    
    def train(self, logger: Logger) -> List[Dict[str, Any]]:
        """
            This can train or perform any long running, offline steps to setup the generation.
        Args:
            logger (Logger): To log any pertinent information.

        Returns:
            List[Dict[str, Any]]: a list of dictionaries that contains results from training. 
                Should be the same as the logger data.
        """                
        return []

    def generate_level(self) -> Level:
        """Generates a level. Each call of this should generate a different level

        Returns:
            Level: [description]
        """

        raise NotImplementedError("Not implemented in Base class")
    