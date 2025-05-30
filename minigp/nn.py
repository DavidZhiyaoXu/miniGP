import jax.numpy as jnp
import jax
import jax.random as random

from abc import ABC, abstractmethod
from functools import partial
# from dataclasses import dataclass
from flax.struct import dataclass
from typing import Tuple, NamedTuple, Optional, Any, Callable, Dict, List

@dataclass
class AbstractNetState(ABC):
    layer_dims: List[int]

@dataclass
class AbstractNetParameters:
    params: Dict[str, jnp.ndarray]

@dataclass
class SimpleNetState(AbstractNetState):
    pass

@dataclass
class SimpleNetParameters(AbstractNetParameters):
    pass

class AbstractNet(ABC):
    @staticmethod
    @abstractmethod
    def initialize_params(key: random.PRNGKey, layer_dims: list) -> AbstractNetParameters:
        raise NotImplementedError
    

class SimpleNet(AbstractNet):
    @staticmethod
    def initialize_params(key: random.PRNGKey, state: SimpleNetState) -> SimpleNetParameters:
        params = {}
        keys = random.split(key, len(state.layer_dims) - 1)
        for i in range(len(state.layer_dims) - 1):
            W_key, b_key = keys[i], keys[i]
            params[f'W{i+1}'] = random.normal(W_key, (state.layer_dims[i], state.layer_dims[i + 1])) * 0.1 + 1
            params[f'b{i+1}'] = jnp.zeros(state.layer_dims[i + 1]) + 1
        return SimpleNetParameters(params=params)

    @staticmethod
    @jax.jit
    @partial(jnp.vectorize, signature="(n)->()", excluded=(1, 2))
    def forward(X: jnp.ndarray, state: SimpleNetState, params: SimpleNetParameters) -> jnp.ndarray:
        z = X
        num_layers = len(state.layer_dims) - 1
        for i in range(num_layers):
            W, b = params.params[f'W{i+1}'], params.params[f'b{i+1}']
            z = jnp.dot(z, W) + b
            if i < num_layers - 1:
                z = jax.nn.relu(z)
        return z.squeeze()


@dataclass
class ConvexNetState(AbstractNetState):
    pass

@dataclass
class ConvexNetParameters(AbstractNetParameters):
    pass

class ConvexNet(AbstractNet):
    @staticmethod
    def initialize_params(key: random.PRNGKey, state: ConvexNetState) -> ConvexNetParameters:
        params = {}
        num_layers = len(state.layer_dims) - 1
        keys = random.split(key, 2 * num_layers)

        for i in range(num_layers):
            in_dim = state.layer_dims[i]
            out_dim = state.layer_dims[i + 1]

            kW, kU = keys[2 * i], keys[2 * i + 1]
            params[f'raw_W{i+1}'] = 0.1 * random.normal(kW, (in_dim, out_dim))
            params[f'raw_U{i+1}'] = 0.1 * random.normal(kU, (state.layer_dims[0], out_dim))
            params[f'b{i+1}'] = jnp.zeros(out_dim)
        return ConvexNetParameters(params=params)

    @staticmethod
    @jax.jit
    @partial(jnp.vectorize, signature="(n)->()", excluded=(1, 2))
    def forward(x: jnp.ndarray, state: ConvexNetState, params: ConvexNetParameters) -> jnp.ndarray:
        z = x
        num_layers = len(state.layer_dims) - 1
        for i in range(num_layers):
            W = params.params[f'raw_W{i+1}']
            U = params.params[f'raw_U{i+1}']
            b = params.params[f'b{i+1}']
            z = jnp.dot(z, W) + jnp.dot(x, U) + b
            if i < num_layers - 1:
                z = jax.nn.relu(z)
        return z.squeeze()
    
def make_psd(raw_mat: jnp.ndarray) -> jnp.ndarray:
    """
    Convert an unconstrained matrix into a PSD matrix
    via M = raw_mat^T @ raw_mat.
    """
    return raw_mat.T @ raw_mat

class MonotoneOperatorNet:
    """
    A simplified R^n -> R^n monotone operator network that stacks
    multiple "PSD-based linear" layers.

    Each layer i:
      out = out + (W_psd @ out) + b
    with W_psd = raw_W^T @ raw_W ensuring PSD => partial monotonicity.
    """

    def __init__(self, layer_dims: List[int], key: jax.random.PRNGKey):
        self.layer_dims = layer_dims
        self.params = self.initialize_params(layer_dims, key)

    def initialize_params(self, layer_dims: List[int], key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        params = {}
        num_layers = len(layer_dims) - 1
        keys = random.split(key, 2 * num_layers)

        for i in range(num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i+1]
            kW, kB = keys[2*i], keys[2*i + 1]

            # raw_W{i}: unconstrained
            # We'll keep them small so we don't get huge PSDs
            params[f'raw_W{i+1}'] = 0.01 * random.normal(kW, (in_dim, out_dim))

            # bias
            # params[f'b{i}'] = 0.01 * random.normal(kB, (out_dim))
            params[f'b{i+1}'] = jnp.zeros(out_dim)

        return params

    def __call__(self, X: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Forward pass of the monotone operator network.
        """
        out = X
        num_layers = len(self.layer_dims) - 1

        for i in range(num_layers):
            raw_W = params[f'raw_W{i+1}']
            b     = params[f'b{i+1}']

            if i < num_layers - 1 and i > 0:
                W_psd = make_psd(raw_W)
            else:
                W_psd = raw_W
            out = out + jnp.dot(out, W_psd) + b

        return out