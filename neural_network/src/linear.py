from abc import ABC, abstractmethod
import numpy as np


# numpy reshape
class linearLayer(ABC):
    @abstractmethod
    def __init__(self, W, B):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backboard(self, error):
        pass

    @abstractmethod
    def update(self, alpha):
        pass
