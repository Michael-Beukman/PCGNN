import os
from typing import Any, Dict
import pickle
import json
import hashlib
class Config:
    """
        This represents a single experiment instance. It has all the useful details to be able to replicate the results.
    """

    def __init__(self, name: str, game: str, method: str,
                 seed: int, results_directory: str,
                 method_parameters: Dict[str, Any],
                 date: str,
                 project: str = 'NoveltyNeatPCG'):
        """
        Args:
            name (str): Friendly name of this experiment.
            game (str): What game (maze or mario)
            method (str): What method (GeneticAlg, NoveltyNeat, PCGRL, etc)
            seed (int): The random seed that was used.
            results_directory (str): Where the results, like pickles, levels, etc are stored.
            method_parameters (Dict[str, Any]): Any other parameters that the method requires. 
                Things like pop_size for a GA, etc.
        """
        self.name = name
        self.game = game
        self.method = method
        self.seed = seed
        self.results_directory = results_directory
        self.method_parameters = method_parameters
        self.date = date
        self.project = project

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary that can be sent to wandb that contains all the important parameters.

        Returns:
            Dict[str, Any]: [description]
        """
        return {
            'game': self.game,
            'method': self.method,
            'seed': self.seed,
            'results_directory': self.results_directory,
            'method_parameters': self.method_parameters,
            'date': self.date
        }

    def hash(self, seed=False) -> int:
        s = self.name + self.game + self.method + self.date + self.params_to_pretty()
        if seed:
            s += str(seed)
        return f"{self.name}-{self.game}-{self.method}-" + str(hashlib.md5(s.encode('utf-8')).hexdigest())
    
    def params_to_pretty(self):
        return '_'.join(f'{key}_{value}' for key, value in self.method_parameters.items())

        pass

    def to_file(self) -> str:
        """Writes to a file and returns the filename

        Returns:
            str: Filename
        """
        location = f'configs/{self.name}/{self.method}/{self.date}/'
        os.makedirs(location, exist_ok=True)
        name = os.path.join(location, f'{self.hash().split("-")[-1][:6]}_{self.params_to_pretty()}_{self.seed}_config.json')
        dic = self.to_dict()
        dic['name'] = self.name
        with open(name, 'w+') as f:
            json.dump(dic, f)
        return name
    
    @staticmethod
    def from_file(filename) -> "Config":
        """Reads the condig from a file
        Returns:
            Config
        """
        with open(filename, 'r') as f:
            dic = json.load(f)
        return Config(**dic)