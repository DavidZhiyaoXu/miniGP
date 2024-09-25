import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from .kernel import AbstractKernel


class GaussianProcess():
    def __init__(self, kernel: AbstractKernel, noise: float = 0):
        self.kernel = kernel
        self.X_train = None
        self.y_train = None
        self.K_fact = None
    
    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.K_fact = torch.linalg.cholesky(self.kernel.matrix(self.X_train, self.X_train))

    # TODO: Should maintain a factorization of K_inverse. e.g. Cholesky + Schur Complement
    def update(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        L11 = self.K_fact
        A21 = self.kernel.matrix(X_train, self.X_train)
        A22 = self.kernel.matrix(X_train, X_train)
        L21 = torch.linalg.solve(L11,A21.T).T
        L22 = torch.linalg.cholesky(A22 - A21 @ torch.cholesky_solve(A21.T,L11))
        L_left = torch.cat([L11, L21], dim=0)
        L_right = torch.cat([torch.zeros_like(L21.T), L22], dim=0)
        self.K_fact = torch.cat([L_left, L_right], dim=1)
        self.X_train = torch.cat([self.X_train, X_train], dim=0)
        self.y_train = torch.cat([self.y_train, y_train], dim=0)

    def predict(self, X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        K_s = self.kernel.matrix(self.X_train, X_test)
        K_ss = self.kernel.matrix(X_test, X_test)

        KsT_K_inv = torch.cholesky_solve(K_s,self.K_fact).T
        
        mu_s = KsT_K_inv @ self.y_train
        cov_s = K_ss - KsT_K_inv @ K_s

        return mu_s, cov_s