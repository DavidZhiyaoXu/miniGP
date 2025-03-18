import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, NamedTuple, Optional, Any, Callable, Dict, List


def plot_process(X_train: jnp.ndarray, y_train: jnp.ndarray, X_test: jnp.ndarray, mu_s: jnp.ndarray, std_s: jnp.ndarray, y_true: Optional[jnp.ndarray] = None) -> None:
    X_test_np = np.array(X_test)
    mu_s_np = np.array(mu_s).flatten()
    std_s_np = np.array(std_s)

    plt.plot(X_test_np, mu_s_np, 'b-', label="Predicted Mean")
    plt.fill_between(X_test_np.flatten(),
                        (mu_s_np - 1.96 * std_s_np).flatten(),
                        (mu_s_np + 1.96 * std_s_np).flatten(),
                        color='blue', alpha=0.2, label="95% Confidence Interval")

    if y_true is not None:
        plt.plot(np.array(X_test), np.array(y_true), color='orangered', linestyle='--', alpha=0.7, label="True Data")

    if X_train is not None and y_train is not None:
        plt.plot(np.array(X_train), np.array(y_train), color='orangered', marker='o', alpha=0.7, label="Training Data")
    plt.title("Gaussian Process Regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

def plot_diffeomorphism_1D(X_test: jnp.ndarray, kernel, kernel_params) -> None:
    X_transformed = kernel.net_fn(X_test, kernel_params.nn_params)

    X_test_np = np.array(X_test)
    X_transformed_np = np.array(X_transformed)

    print(X_transformed_np)

    np.set_printoptions(suppress=True, precision=3)
    # print(X_test_np.flatten())
    # print(X_transformed_np.flatten())
    # print((X_test_np / X_transformed_np).flatten())

    plt.plot(X_test_np, X_transformed_np, label="Warped Space")
    # Plot the scatter points of diffeomorphism.
    # plt.scatter(X_transformed_np[:, 0], X_transformed_np[:, 1], color='white', edgecolor='black', marker='o', label="Transformed Points", alpha=0.7)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Diffeomorphism Learned")
    plt.legend()
    plt.show()