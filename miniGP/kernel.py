import torch
from abc import ABC, abstractmethod


class AbstractKernel(ABC):
    @abstractmethod
    def matrix(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        pass


class RBFKernel(AbstractKernel):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
    
    def matrix(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(X1, X2)
        return torch.exp(- dist**2 / (2 * self.sigma**2))