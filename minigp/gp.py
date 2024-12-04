import torch

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp
import optax

from typing import Tuple, NamedTuple, Optional, Any, Callable, Dict
from dataclasses import dataclass

from .kernel import AbstractKernel, AbstractKernelParameters, DeepKernel, DeepKernelParameters


@dataclass
class GaussianProcessState():
    X_train: jnp.ndarray
    y_train: jnp.ndarray
    K: jnp.ndarray
    L: jnp.ndarray
    kernel: AbstractKernel

@dataclass
class GaussianProcessParameters():
    noise: float = jnp.log(1e-6)

class GaussianProcess():
    def init_state_with_params(self,
                               kernel: AbstractKernel,
                               kernel_params: AbstractKernelParameters,
                               X_train: jnp.ndarray,
                               y_train: jnp.ndarray) -> Tuple[GaussianProcessState, GaussianProcessParameters]:
        params = GaussianProcessParameters(noise = jnp.log(1e-6)) 
        K = kernel.matrix(X_train, X_train, kernel_params) + jnp.exp(params.noise) * jnp.eye(len(X_train))
        L = jnp.linalg.cholesky(K)
        state = GaussianProcessState(X_train=X_train, y_train=y_train, K=K, L=L, kernel=kernel)
        return (state, params)
    
    def fit(self, state: GaussianProcessState, kernel_params: AbstractKernelParameters, gp_params: GaussianProcessParameters, jitter: float = 1e-6) -> Tuple[GaussianProcessState, GaussianProcessParameters]:
        params = gp_params
        K = state.kernel.matrix(state.X_train, state.X_train, kernel_params) + (jnp.exp(gp_params.noise) + jitter) * jnp.eye(len(state.X_train))
        L = jnp.linalg.cholesky(K)
        state = GaussianProcessState(X_train=state.X_train, y_train=state.y_train, K=K, L=L, kernel=state.kernel)
        return (state, params)
    
    def predict(self, state: GaussianProcessState, kernel_params: AbstractKernelParameters, gp_params: GaussianProcessParameters, X_test: jnp.ndarray):
        """Predict the mean and variance at test points."""
        K_s = state.kernel.matrix(state.X_train, X_test, kernel_params)
        K_ss = state.kernel.matrix(X_test, X_test, kernel_params) + jnp.exp(gp_params.noise) * jnp.eye(len(X_test))

        # Solve for KsT_K_inv
        KsT_K_inv = jsp.linalg.solve_triangular(state.L, K_s, lower=True)
        KsT_K_inv = jsp.linalg.solve_triangular(state.L.T, KsT_K_inv, lower=False)
        KsT_K_inv = KsT_K_inv.T

        mu_s = KsT_K_inv @ state.y_train
        cov_s = K_ss - KsT_K_inv @ K_s

        return mu_s, jnp.diag(cov_s)

    def log_marginal_likelihood(self, state: GaussianProcessState, kernel_params: AbstractKernelParameters, gp_params: GaussianProcessParameters) -> float:
        K = state.kernel.matrix(state.X_train, state.X_train, kernel_params) + jnp.exp(gp_params.noise) * jnp.eye(len(state.X_train))
        L = jnp.linalg.cholesky(K)

        alpha = jsp.linalg.solve_triangular(L, state.y_train, lower=True)
        alpha = jsp.linalg.solve_triangular(L.T, alpha, lower=False)

        log_likelihood = -0.5 * jnp.dot(state.y_train, alpha)
        log_likelihood -= jnp.sum(jnp.log(jnp.diag(L))) / 2
        log_likelihood -= 0.5 * len(state.y_train) * jnp.log(2 * jnp.pi)
        return log_likelihood



def optimize_mle_nn(gp: GaussianProcess, state: GaussianProcessState, kernel_params: DeepKernelParameters, gp_params: GaussianProcessParameters, num_iters: int = 200, learning_rate: float = 0.01):
    def objective(params):
        kernel_params = DeepKernelParameters(log_alpha=params['log_alpha'], nn_params=params['nn_params'])
        gp_params = GaussianProcessParameters(noise=params['noise'])
        return -gp.log_marginal_likelihood(state, kernel_params, gp_params)

    # Flatten the parameters for optimization, NN kernel specific.
    params = {
        'log_alpha': kernel_params.log_alpha,
        'nn_params': kernel_params.nn_params,
        'noise': gp_params.noise
        }
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Optimization loop
    @jax.jit
    def step(opt_state, params):
        loss, grads = jax.value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(grads, opt_state,params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    for i in range(num_iters):
        opt_state, params, loss = step(opt_state, params)
        if (i + 1) % 50 == 0:
            print(f"Iteration {i+1}, Loss: {loss:.4f}")

    optimized_kernel_params = DeepKernelParameters(log_alpha=params['log_alpha'], nn_params=params['nn_params'])
    optimized_gp_params = GaussianProcessParameters(noise=params['noise'])

    return optimized_kernel_params, optimized_gp_params