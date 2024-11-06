import torch

import jax
from jax import grad, jit, value_and_grad
import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp

from abc import ABC, abstractmethod
from typing import Tuple, NamedTuple, Optional, Any, Callable, Dict
from dataclasses import dataclass

from .utils import AbstractNN

@dataclass
class AbstractKernelParameters(ABC):
    @abstractmethod
    def param_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


class AbstractKernel(ABC):
    @abstractmethod
    def init_params(self):
        raise NotImplementedError

    @abstractmethod
    def matrix(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


@dataclass
class DeepKernelParameters(AbstractKernelParameters):
    sigma: float
    nn_params: Dict[str, Any]
    def param_dict(self) -> Dict[str, Any]:
        return {
            "sigma": self.sigma,
            "nn_params": self.nn_params
        }

class DeepKernel(AbstractKernel):
    def __init__(
        self,
        net_fn: AbstractNN,
        layer_dims: list
    ):
        self.net_fn = net_fn
        self.layer_dims = layer_dims

    def init_params(self, key: random.PRNGKey) -> DeepKernelParameters:
        nn_model = self.net_fn
        nn_params = nn_model.params  
        return DeepKernelParameters(
            sigma = 1.0,
            nn_params = nn_params
        )


    def matrix(self, X1: jnp.ndarray, X2: jnp.ndarray, params: DeepKernelParameters) -> jnp.ndarray:
        nn_model = self.net_fn
        X1_expanded = nn_model(X1, params.nn_params)
        X2_expanded = nn_model(X2, params.nn_params)
        sqdist = jnp.sum((X1_expanded[:, None] - X2_expanded[None, :]) ** 2, axis=2)
        return jnp.exp(-0.5 * sqdist / (params.sigma ** 2))

        

# class HeightKernel(AbstractKernel):
#     def init_params(self):
#         pass


# class HeightKernelParameters(AbstractKernelParameters):
#     def param_dict(self) -> Dict[str, Any]:
#         raise NotImplementedError


# class AbstractKernel(ABC):
#     @abstractmethod
#     def matrix(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
#         pass


# class RBFKernel(AbstractKernel):
#     def __init__(self, sigma: float = 1.0):
#         super().__init__()
#         self.sigma = sigma
    
#     def matrix(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
#         dist = torch.cdist(X1, X2)
#         return torch.exp(- dist**2 / (2 * self.sigma**2))