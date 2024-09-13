import torch
from typing import Callable
from scipy.stats import norm
from abc import ABC, abstractmethod


class AbstractAcquisition(ABC):
    """
    Abstract quisition function class for minimization.
    """
    @abstractmethod
    def evaluate(self, X: torch.Tensor, y_best: float) -> torch.Tensor:
        pass


class PI(AbstractAcquisition):
    def __init__(self, gp_model: Callable):
        self.gp_model = gp_model
    
    def evaluate(self, X: torch.Tensor, y_best: float) -> torch.Tensor:
        mu_s, cov_s = self.gp_model.predict(X)
        sigma_s = torch.sqrt(torch.diag(cov_s)).clamp(min=1e-6)
        Z = (y_best - mu_s) / sigma_s
        pi = norm.cdf(Z)
        return -pi

  
class EI(AbstractAcquisition):
    def __init__(self, gp_model: Callable):
        self.gp_model = gp_model
    
    def evaluate(self, X: torch.Tensor, y_best: float) -> torch.Tensor:
        mu_s, cov_s = self.gp_model.predict(X)
        sigma_s = torch.sqrt(torch.diag(cov_s)).clamp(min=1e-6)
        Z = (y_best - mu_s) / sigma_s
        ei = -sigma_s * norm.pdf(Z) - sigma_s * Z * norm.cdf(Z)
        return ei