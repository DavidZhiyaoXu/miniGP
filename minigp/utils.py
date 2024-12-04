import torch

import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Tuple, NamedTuple, Optional, Any, Callable, Dict, List


class AbstractNN(ABC):
    layer_dims : List[int]
    params : Dict[str, jnp.ndarray]
    @abstractmethod
    def initialize_params(self):
        raise NotImplementedError
    

class SimpleNN(AbstractNN):
    layer_dims = None
    params = None
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
            params[f'W{i+1}'] = random.normal(W_key, (layer_dims[i], layer_dims[i + 1])) * 0.1 + 1
            params[f'b{i+1}'] = jnp.zeros(layer_dims[i + 1]) + 1
        # print(params)
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


class ResNetNN(AbstractNN):
    layer_dims = None
    params = None
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
            params[f'W{i+1}'] = random.normal(W_key, (layer_dims[i], layer_dims[i + 1])) * jnp.sqrt(2.0 / (layer_dims[i] + layer_dims[i + 1])) * 0.3 + random.normal(W_key, (layer_dims[i], layer_dims[i + 1])) * jnp.sqrt(2.0 / (layer_dims[i] + layer_dims[i + 1])) * 0.1
            params[f'b{i+1}'] = jnp.zeros(layer_dims[i + 1])
            params[f'b{i+1}'] = random.normal(W_key, (layer_dims[i + 1],)) * 0.01
        return params

    def __call__(self, X: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Forward pass through the ResNet network with residual connections."""
        h = X
        num_layers = len(self.layer_dims) - 1
        # print(f"num_layers = {num_layers}")
        for i in range(0, num_layers, 3):
            W1, b1 = params[f'W{i+1}'], params[f'b{i+1}']
            out = jnp.dot(h, W1) + b1
            out = jax.nn.leaky_relu(out)

            W2, b2 = params[f'W{i+2}'], params[f'b{i+2}']
            out = jnp.dot(out, W2) + b2
            out = jax.nn.leaky_relu(out)

            W3, b3 = params[f'W{i+3}'], params[f'b{i+3}']
            out = jnp.dot(out, W3) + b3

            h = jax.nn.leaky_relu(out) + h
        return h


class Decoder(AbstractNN):
    def __init__():
        raise NotImplementedError



def pretrain_diffeomorphism():
    raise NotImplementedError

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