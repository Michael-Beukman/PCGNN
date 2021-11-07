from typing import Union
import numpy as np
from games.level import Level
from novelty_neat.types import LevelNeuralNet


class NeatLevelGenerator:
    """A superclass for generating levels in different ways.
    """
    def __init__(self, number_of_random_variables: int = 2):
        """
        Args:
            number_of_random_variables (int, optional): The number of random inputs the neural net takes. Defaults to 2.
        """
        self.number_of_random_variables = number_of_random_variables
    
    def __call__(self, net: LevelNeuralNet, input: Union[np.ndarray, None] = None) -> Level:
        """Generates a level from the given network and the input. 

        Args:
            net (LevelNeuralNet): [description]
            input (Union[np.ndarray, None], optional): What to give to the network. If input is None, then we randomly create an input 
                of length self.number_of_random_variables
                Defaults to None.
        Returns:
            Level: [description]
        """
        if input is None: 
            input = np.random.randn(self.number_of_random_variables)
        return self.generate_level(net, input)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.number_of_random_variables})"

    def generate_level(self, net: LevelNeuralNet, input: np.ndarray) -> Level:
        """This should actually generate the level.

        Args:
            net (LevelNeuralNet):
            input (np.ndarray):
        
        Returns:
            Level:
        """
        raise NotImplementedError('')