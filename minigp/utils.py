import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def plot_process(process, X_test: torch.Tensor, y_true: Optional[torch.Tensor] = None) -> None:
    """Plot the GP mean and confidence interval over the test points."""
    mu_s, cov_s = process.predict(X_test)
    std_s = torch.sqrt(torch.diag(cov_s))

    X_test_np = X_test.detach().numpy()
    mu_s_np = mu_s.detach().numpy().flatten()
    std_s_np = std_s.detach().numpy()

    plt.plot(X_test_np, mu_s_np, 'b-', label="Predicted Mean")
    plt.fill_between(X_test_np.flatten(),
                        (mu_s_np - 1.96 * std_s_np).flatten(),
                        (mu_s_np + 1.96 * std_s_np).flatten(),
                        color='blue', alpha=0.2, label="95% Confidence Interval")

    if y_true is not None:
        plt.plot(X_test_np, y_true.detach().numpy(), 'r--', label="True Function")

    if process.X_train is not None and process.y_train is not None:
        plt.scatter(process.X_train.detach().numpy(), process.y_train.detach().numpy(), color='red', zorder=5, label="Training Data")

    plt.title("Gaussian Process Regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

