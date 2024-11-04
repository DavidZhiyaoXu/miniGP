import torch

import jax.numpy as jnp
import jax
from jax import random

import matplotlib.pyplot as plt
from typing import Tuple, NamedTuple, Optional, Any, Callable, Dict


def plot_process(process, X_test: jnp.ndarray, y_true: Optional[jnp.ndarray] = None) -> None:
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

class SimpleNN:
    def __init__(self, layer_dims: list, key: random.PRNGKey):
        """Initialize a simple feed-forward neural network with specified layer dimensions."""
        self.layer_dims = layer_dims
        self.params = self.initialize_params(layer_dims, key)

    def initialize_params(self, layer_dims: list, key: random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Initialize network parameters (weights and biases) for each layer."""
        params = {}
        keys = random.split(key, len(layer_dims) - 1)
        for i in range(len(layer_dims) - 1):
            W_key, b_key = keys[i], keys[i]
            params[f'W{i+1}'] = random.normal(W_key, (layer_dims[i], layer_dims[i + 1])) * 0.1
            params[f'b{i+1}'] = jnp.zeros(layer_dims[i + 1])
        return params

    def __call__(self, X: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Forward pass through the neural network."""
        h = X
        num_layers = len(self.layer_dims) - 1
        for i in range(num_layers):
            W, b = params[f'W{i+1}'], params[f'b{i+1}']
            h = jnp.dot(h, W) + b
            if i < num_layers - 1:  # ReLU activation for all layers except the last
                h = jax.nn.relu(h)
        return h


class ResNetNN:
    def __init__(self, layer_dims: list, key: random.PRNGKey):
        """Initialize a ResNet-style neural network with residual connections."""
        self.layer_dims = layer_dims
        self.params = self.initialize_params(layer_dims, key)

    def initialize_params(self, layer_dims: list, key: random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Initialize parameters for the ResNet-style network."""
        params = {}
        keys = random.split(key, len(layer_dims) - 1)
        for i in range(len(layer_dims) - 1):
            W_key, b_key = keys[i], keys[i]
            params[f'W{i+1}'] = random.normal(W_key, (layer_dims[i], layer_dims[i + 1])) * 0.1
            params[f'b{i+1}'] = jnp.zeros(layer_dims[i + 1])
        return params

    def __call__(self, X: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Forward pass through the ResNet network with residual connections."""
        h = X
        num_layers = len(self.layer_dims) - 1
        for i in range(0, num_layers, 2):
            # First layer in the residual block
            W1, b1 = params[f'W{i+1}'], params[f'b{i+1}']
            out = jnp.dot(h, W1) + b1
            out = jax.nn.relu(out)

            # Second layer in the residual block
            W2, b2 = params[f'W{i+2}'], params[f'b{i+2}']
            out = jnp.dot(out, W2) + b2

            # Add residual connection
            h = jax.nn.relu(out + h)
        return h
