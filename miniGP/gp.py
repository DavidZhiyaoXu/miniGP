import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from .kernel import AbstractKernel


class GaussianProcess():
    def __init__(self, kernel: AbstractKernel, noise: float = 0):
        self.kernel = kernel
        self.X_train = None
        self.y_train = None
        self.K = None
        # TODO: Initialize field storing factorization of K_inverse
    
    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.K = self.kernel.matrix(self.X_train, self.X_train)

    # TODO: Should maintain a factorization of K_inverse. e.g. Cholesky + Schur Complement
    def update(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        pass

    def predict(self, X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        K_s = self.kernel.matrix(self.X_train, X_test)
        K_ss = self.kernel.matrix(X_test, X_test)

        # TODO: Change to efficient evaluation.
        K_inv = torch.inverse(self.K)

        mu_s = K_s.T @ K_inv @ self.y_train
        cov_s = K_ss - K_s.T @ K_inv @ K_s
        
        return mu_s, cov_s

    def plot(self, X_test: torch.Tensor, y_true: Optional[torch.Tensor] = None) -> None:
        """Plot the GP mean and confidence interval over the test points."""
        mu_s, cov_s = self.predict(X_test)
        std_s = torch.sqrt(torch.diag(cov_s))

        X_test_np = X_test.detach().numpy()
        mu_s_np = mu_s.detach().numpy().flatten()
        std_s_np = std_s.detach().numpy()

        plt.plot(X_test_np, mu_s_np, 'b-', label="Predicted Mean")
        plt.fill_between(X_test_np.flatten(),
                         (mu_s_np - 1.96 * std_s_np).flatten(),
                         (mu_s_np + 1.96 * std_s_np).flatten(),
                         color='blue', alpha=0.2, label="95% Confidence Interval")

        # If true values are provided, plot them
        if y_true is not None:
            plt.plot(X_test_np, y_true.detach().numpy(), 'r--', label="True Function")

        # Scatter plot of training data
        if self.X_train is not None and self.y_train is not None:
            plt.scatter(self.X_train.detach().numpy(), self.y_train.detach().numpy(), color='red', zorder=5, label="Training Data")

        plt.title("Gaussian Process Regression")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()