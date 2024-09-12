import numpy as np
import torch
from typing import Tuple, Callable


class GaussianProcess:
    def __init__(self, kernel: Callable, noise: float = 0):
        self.kernel = kernel
        self.X_train = None
        self.y_train = None
        self.K = None
        # TODO: Initialize field storing factorization of K_inverse
    
    def fit(self, X_train: torch.tensor, y_train: torch.tensor) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.K = self.kernel(self.X_train, self.X_train)

    # TODO: Should maintain a factorization of K_inverse. e.g. Cholesky + Schur Complement
    def update(self, X_train: torch.tensor, y_train: torch.tensor) -> None:
        pass

    def predict(self, X_test: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        X_test_torch = torch.tensor(X_test)
        K_s = self.kernel(self.X_train, X_test_torch)
        K_ss = self.kernel(X_test_torch, X_test_torch)

        # Inefficient computation of inverse. J
        K_inv = torch.inverse(self.K)

        mu_s = K_s.T @ K_inv @ self.y_train
        cov_s = K_ss - K_s.T @ K_inv @ K_s
        
        return mu_s, cov_s
