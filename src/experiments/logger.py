import os
from typing import Any, Dict
import wandb
from common.types import Verbosity
from common.utils import save_compressed_pickle

from experiments.config import Config


class Logger:
    """
        A basic logger that acts as the interface to log results
    """
    def __init__(self, config: Config, verbose: Verbosity = Verbosity.NONE, seed: int = 0) -> None:
        self.config = config
        self.verbose = verbose
        self.LOG_ALL_FITNESSES = False
        self.seed = seed
    
    def log(self, dic: Dict[Any, Any], **kwargs):
        """Logs this dictionary

        Args:
            dic (Dict[Any, Any]): 
        """
        pass

    def end(self, name: str, seed: int):
        pass


class WandBLogger(Logger):
    """Logs everything to weights and biases (https://wandb.ai/)
    """
    def __init__(self, config: Config, verbose: Verbosity = Verbosity.NONE, seed: int = 0) -> None:
        super().__init__(config, verbose, seed)
        wandb.init(
            project=config.project,
            notes = config.name,
            tags = [config.game, config.method],
            config=config.to_dict(),
            group=config.hash(seed=False),
            job_type="run"
        )
        
    def log(self, dic: Dict[Any, Any], **kwargs):
        wandb.log(dic, **kwargs)


class NoneLogger(Logger):
    """
        A logger that is basically a no-op. Kind of null object pattern
    """
    def __init__(self, seed: int = 0) -> None:
        super().__init__(None, verbose=Verbosity.NONE, seed=seed)

class PrintLogger(Logger):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(None, verbose=Verbosity.DETAILED, seed=seed)
    def log(self, dic: Dict[Any, Any], **kwargs):
        print(dic)
        
        
class WriteToDictionaryLogger:
    """
        Logs to a dictionary. Only used for PCGGN fitness
    """
    def __init__(self, config: Config, verbose: Verbosity = Verbosity.NONE, seed: int = 0) -> None:
        self.config = config
        self.verbose = verbose
        self.LOG_ALL_FITNESSES = True
        
        self.list_of_things = []
        self.seed = seed
    
    def log(self, dic: Dict[Any, Any], **kwargs):
        """Logs this dictionary

        Args:
            dic (Dict[Any, Any]): 
        """
        self.list_of_things.append(dic)

    def end(self, name: str, seed:int):
        # f
        D = f"results/all_models/pcgnn/{name}/logs/"
        os.makedirs(D, exist_ok=True)
        save_compressed_pickle(os.path.join(D, f'{seed}_all_logs'), self.list_of_things)
        del self.list_of_things
        self.list_of_things = []