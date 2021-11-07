from typing import Any, Dict
import wandb
from common.types import Verbosity

from experiments.config import Config


class Logger:
    """
        A basic logger that acts as the interface to log results
    """
    def __init__(self, config: Config, verbose: Verbosity = Verbosity.NONE) -> None:
        self.config = config
        self.verbose = verbose
    def log(self, dic: Dict[Any, Any], **kwargs):
        """Logs this dictionary

        Args:
            dic (Dict[Any, Any]): 
        """
        pass


class WandBLogger(Logger):
    """Logs everything to weights and biases (https://wandb.ai/)
    """
    def __init__(self, config: Config, verbose: Verbosity = Verbosity.NONE) -> None:
        super().__init__(config, verbose)
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
    def __init__(self) -> None:
        super().__init__(None, verbose=Verbosity.NONE)

class PrintLogger(Logger):
    def __init__(self) -> None:
        super().__init__(None, verbose=Verbosity.DETAILED)
    def log(self, dic: Dict[Any, Any], **kwargs):
        print(dic)