import torch
from typing import Tuple, Callable
from .gp import GaussianProcess
from .acquisition import AbstractAcquisition
from .utils import *

class BayesianOptimizer():
    def __init__(self, objective: Callable, gp_model: GaussianProcess, acq_func: AbstractAcquisition, dims: int, bounds: torch.Tensor):
        # TODO: Check dims and bounds match.
        self.objective = objective
        self.gp_model = gp_model
        self.acq_func = acq_func
        self.dims = dims
        self.bounds = bounds
        self.y_best = float('inf')

    def optimize(self, n_warmup: int, n_iter: int) -> torch.Tensor:
        X_train = torch.rand((n_warmup, self.bounds.shape[0])) * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        self.gp_model.fit(X_train, self.objective(X_train))
        self.y_best = torch.min(self.gp_model.y_train)
        grid = torch.linspace(0, 10, 100).unsqueeze(1)
        for i in range(n_iter):
            X_next = self.select_next()
            self.gp_model.update(X_next.unsqueeze(0), self.objective(X_next.unsqueeze(0)))
            self.y_best = torch.min(self.gp_model.y_train)
            plot_process(self.gp_model, grid)
    
    def select_next(self) -> torch.Tensor:
        X_candidate = torch.rand((100, self.bounds.shape[0])) * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        scores = self.acq_func.evaluate(X_candidate, self.y_best)
        return X_candidate[torch.argmin(scores)]
    
    # TODO: Should call utils
    def plot_gif() -> None:
        pass