import jax
from jax import grad, jit, value_and_grad
import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp

from abc import ABC, abstractmethod
from typing import Tuple, NamedTuple, Optional, Any, Callable, Dict
from dataclasses import dataclass

from .warpednn import AbstractNN

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
class GaussianKernelParameters(AbstractKernelParameters):
    log_alpha: float
    sigma: float
    def param_dict(self) -> Dict[str, Any]:
        return {
            "log_alpha": self.log_alpha,
            "sigma": self.sigma
        }


class GaussianKernel(AbstractKernel):
    def init_params(self, key: random.PRNGKey, log_alpha: float = -5.0, sigma: float = 1.0) -> GaussianKernelParameters:
        return GaussianKernelParameters(
            log_alpha = log_alpha,
            sigma = sigma
        )


    def matrix(self, X1: jnp.ndarray, X2: jnp.ndarray, params: GaussianKernelParameters) -> jnp.ndarray:
        sqdist = jnp.sum((X1[:, None] - X2[None, :]) ** 2, axis=2)
        return jnp.exp(params.log_alpha) * jnp.exp(-0.5 * sqdist / (params.sigma ** 2) )
        # return jnp.exp(-0.5 * sqdist / (params.sigma ** 2) )



@dataclass
class DeepKernelParameters(AbstractKernelParameters):
    log_alpha: float
    nn_params: Dict[str, Any]
    def param_dict(self) -> Dict[str, Any]:
        return {
            "log_alpha": self.log_alpha,
            "nn_params": self.nn_params
        }

class DeepKernel(AbstractKernel):
    def __init__(
        self,
        net_fn: AbstractNN
    ):
        self.net_fn = net_fn

    def init_params(self, key: random.PRNGKey, log_alpha: float = -5.0) -> DeepKernelParameters:
        nn_model = self.net_fn
        nn_params = nn_model.params
        return DeepKernelParameters(
            log_alpha = log_alpha,
            nn_params = nn_params
        )


    def matrix(self, X1: jnp.ndarray, X2: jnp.ndarray, params: DeepKernelParameters) -> jnp.ndarray:
        nn_model = self.net_fn
        X1_expanded = nn_model(X1, params.nn_params)
        X2_expanded = nn_model(X2, params.nn_params)
        sqdist = jnp.sum((X1_expanded[:, None] - X2_expanded[None, :]) ** 2, axis=2)
        return jnp.exp(params.log_alpha) * jnp.exp(-0.5 * sqdist )

        
@dataclass
class HeightKernelParameters(AbstractKernelParameters):
    log_alpha: float
    sigma: float
    coef: jnp.ndarray
    def param_dict(self) -> Dict[str, Any]:
        return {
            "log_alpha": self.log_alpha,
            "sigma": self.sigma,
            "coef": self.coef
        }


class HeightKernel(AbstractKernel):
    def __init__(
        self,
        degree: int
    ):
        self.degree = degree

    def init_params(self, key: random.PRNGKey, log_alpha: float = -5.0, sigma: float = 3.0) -> GaussianKernelParameters:
        return HeightKernelParameters(
            log_alpha = log_alpha,
            sigma = sigma,
            coef = 1e-2 * random.normal(key, shape=(self.degree,))
        )

    def height(self, X: jnp.ndarray, coef: jnp.ndarray) -> jnp.ndarray:
        terms = [X**d for d in range(self.degree)]
        terms = jnp.stack(terms)
        return jnp.dot(coef, terms)
    
    def height_batch(self, X: jnp.ndarray, coef: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda x: self.height(x, coef))(X)

    def matrix(self, X1: jnp.ndarray, X2: jnp.ndarray, params: GaussianKernelParameters) -> jnp.ndarray:
        h_X1 = self.height_batch(X1, params.coef)
        h_X1 = jnp.squeeze(h_X1, axis=-1)
        h_X2 = self.height_batch(X2, params.coef)
        h_X2 = jnp.squeeze(h_X2, axis=-1)
        X1_expanded = jnp.concatenate([X1, h_X1[:, None]], axis=1)
        X2_expanded = jnp.concatenate([X2, h_X2[:, None]], axis=1)
        sqdist = jnp.sum((X1_expanded[:, None] - X2_expanded[None, :]) ** 2, axis=2)
        return jnp.exp(params.log_alpha) * jnp.exp(-0.5 * sqdist / (params.sigma ** 2) )
        # return jnp.exp(-0.5 * sqdist / (params.sigma ** 2) )