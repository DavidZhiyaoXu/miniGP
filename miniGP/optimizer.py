import torch
from typing import Tuple, Callable
from .gp import GaussianProcess
from .acquisition import AbstractAcquisition

class BayesianOptimization():
    def __init__(self, objective: Callable, gp_model: GaussianProcess, acq_func: AbstractAcquisition, dims: int, bounds: torch.Tensor):
        # TODO: Check dims and bounds match.
        self.objective = objective
        self.gp_model = gp_model
        self.acq_func = acq_func
        self.dims = dims
        self.bounds = bounds
        self.y_best = float('inf')

    def optimize(self: int, n_warmup: int, n_iter) -> torch.Tensor:
        X_train = torch.rand((n_warmup, self.bounds.shape[0])) * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        self.gp_model.fit(X_train, self.objective(X_train))
        self.y_best = torch.min(self.gp_model.y_train)
        for i in range(n_iter):
            X_next = self.select_next()
            X_train = torch.cat((X_train, X_next.unsqueeze(0)), dim=0)
            y_train = torch.cat((self.gp_model.y_train, self.objective(X_next.unsqueeze(0)).unsqueeze(0)), dim=0)
            self.gp_model.fit(X_train, y_train)
            self.y_best = torch.min(self.gp_model.y_train)
        return self.y_best
    
    def select_next() -> torch.Tensor:
        # TODO: Optimize acquisition function
        pass
    
    # TODO: Should call gp.plot
    def plot() -> None:
        pass