import sys
import logging
from torch_geometric import seed_everything
import numpy as np


class Config():
    def __init__(self, seed=42, **kwargs):
        """
        Configuration class for the project to store hyperparameters and other settings.
        """
        self.args = " ".join(sys.argv)
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.seed = seed

    def seed_all(self):
        """
        Seed everything
        """
        seed_everything(self.seed)
        self.rng = np.random.RandomState(self.seed)

    def add(self, **kwargs):
        """
        Add a new attribute to the configuration
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        """
        Print the configuration
        """
        l_k = min(max([len(k) for k in self.__dict__.keys()]), 25) + 1
        s = ""
        for k, v in sorted(self.__dict__.items()):
            s += f"{k.ljust(l_k)}: {v}" + "\n"
        return s

    def __str__(self):
        """
        Print the configuration
        """
        l_k = min(max([len(k) for k in self.__dict__.keys()]), 25) + 1
        s = ""
        for k, v in sorted(self.__dict__.items()):
            s += f"{k.ljust(l_k)}: {v}" + "\n"
        return s

    def __getitem__(self, key):
        """
        Get an item from the configuration
        """
        return self.__dict__[key]

    def __setitem__(self, key, value):
        """
        Set an item in the configuration
        """
        self.__dict__[key] = value

    def __contains__(self, key):
        """
        Check if an item is in the configuration
        """
        return key in self.__dict__

    def __iter__(self):
        """
        Iterate over the configuration
        """
        return iter(self.__dict__)

    def keys(self):
        """
        Get the keys of the configuration
        """
        return self.__dict__.keys()

    def values(self):
        """
        Get the values of the configuration
        """
        return self.__dict__.values()

    def items(self):
        """
        Get the items of the configuration
        """
        return self.__dict__.items()

    def sumup(self, show=False, write=True):
        """
        Write the configuration to log file.
        """
        path = self.UNIQUE
        file_name = f"{path}/config.txt"
        if write:
            with open(file_name, "w") as f:
                f.write(str(self))
        if show:
            cfg.logger.info(str(self))
            if write:
                cfg.logger.info(f"Configuration saved to {file_name}")



cfg = Config()
