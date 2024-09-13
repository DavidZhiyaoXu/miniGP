import torch
from typing import Tuple, Callable
from .gp import GaussianProcess
from .acquisition import AbstractAcquisition

class BayesianOptimization():
    def __init__(self, objective: Callable, gp_model: GaussianProcess, acq_func: AbstractAcquisition, dims: int, bounds: torch.tensor):
        # TODO: Check dims and bounds match.
        self.gp_model = gp_model
        self.acq_func = acq_func
        self.dims = dims
        self.bounds = bounds
        self.y_best = float('inf')

    def optimize(self: int, n_warmup: int, n_iter) -> torch.tensor:
        X_train = torch.rand((n_warmup, self.bounds.shape[0])) * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        self.gp_model.fit(X_train, self.sample_objective(X_train))
        for i in range(n_iter):
            self.gp_model.fit(X_train, self.sample_objective(X_train))
            self.y_best = torch.min(self.sample_objective(X_train))
            X_next = self.propose_location(self.acquisition_func, self.y_best)
            X_train = torch.cat((X_train, X_next.unsqueeze(0)), dim=0)
        return self.y_best
    
    # TODO: Should call gp.plot
    def plot() -> None:
        pass