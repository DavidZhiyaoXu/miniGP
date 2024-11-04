import torch

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp
import optax

from typing import Tuple, NamedTuple, Optional, Any, Callable, Dict
from dataclasses import dataclass

from .kernel import AbstractKernel, AbstractKernelParameters


@dataclass
class GaussianProcessState():
    X_train: jnp.ndarray
    y_train: jnp.ndarray
    K: jnp.ndarray
    L: jnp.ndarray
    kernel: AbstractKernel

@dataclass
class GaussianProcessParameters():
    noise: float = 1e-6

class GaussianProcess():
    def init_params_with_state(self,
                               kernel: AbstractKernel,
                               kernel_params: AbstractKernelParameters,
                               gp_params: GaussianProcessParameters,
                               X_train: jnp.ndarray,
                               y_train: jnp.ndarray):
        K = kernel(X_train, X_train, kernel_params) + gp_params.noise * jnp.eye(len(X_train))
        L = jnp.linalg.cholesky(K)
        states = GaussianProcessState(X_train=X_train, y_train=y_train, K=K, L=L, kernel=kernel)
        params = GaussianProcessParameters(noise = 1e-6) 
        return (params, states)
    
    def predict(self, state: GaussianProcessState, kernel_params: AbstractKernelParameters, gp_params: GaussianProcessParameters, X_test: jnp.ndarray):
        """Predict the mean and variance at test points."""
        K_s = state.kernel(state.X_train, X_test, kernel_params)
        K_ss = state.kernel(X_test, X_test, kernel_params) + gp_params.noise * jnp.eye(len(X_test))

        # Solve for alpha
        alpha = jsp.linalg.solve_triangular(state.L, state.y_train, lower=True)
        alpha = jsp.linalg.solve_triangular(state.L.T, alpha, lower=False)

        # Predictive mean
        mu_s = K_s.T @ alpha

        # Predictive variance
        v = jsp.linalg.solve_triangular(state.L, K_s, lower=True)
        cov_s = K_ss - v.T @ v
        return mu_s, jnp.diag(cov_s)

    def log_marginal_likelihood(self, state: GaussianProcessState, kernel_params: AbstractKernelParameters, gp_params: GaussianProcessParameters) -> float:
        K = state.kernel(state.X_train, state.X_train, kernel_params) + gp_params.noise * jnp.eye(len(state.X_train))
        L = jnp.linalg.cholesky(K)

        alpha = jsp.linalg.solve_triangular(L, state.y_train, lower=True)
        alpha = jsp.linalg.solve_triangular(L.T, alpha, lower=False)

        log_likelihood = -0.5 * jnp.dot(state.y_train, alpha)
        log_likelihood -= jnp.sum(jnp.log(jnp.diag(L)))
        log_likelihood -= 0.5 * len(state.y_train) * jnp.log(2 * jnp.pi)
        return log_likelihood



def optimize_mle(gp: GaussianProcess, state: GaussianProcessState, kernel_params: ResNetKernelParameters, gp_params: GaussianProcessParameters, num_iters: int = 100, learning_rate: float = 0.01):
    """Optimize GP and kernel parameters by maximizing the log marginal likelihood."""
    
    # Define the objective function
    def objective(params):
        kernel_params, gp_params = params
        return -gp.log_marginal_likelihood(state, kernel_params, gp_params)

    # Flatten the parameters for optimization
    params = (kernel_params, gp_params)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Optimization loop
    @jax.jit
    def step(opt_state, params):
        loss, grads = jax.value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    for i in range(num_iters):
        opt_state, params, loss = step(opt_state, params)
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}, Loss: {loss:.4f}")

    # Unpack optimized parameters
    return params


# class GaussianProcess():
#     def __init__(self, kernel: AbstractKernel, noise: float = 0):
#         self.kernel = kernel
#         self.X_train = None
#         self.y_train = None
#         self.K_fact = None
    
#     def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
#         self.X_train = X_train
#         self.y_train = y_train
#         self.K_fact = torch.linalg.cholesky(self.kernel.matrix(self.X_train, self.X_train))

#     # TODO: Should maintain a factorization of K_inverse. e.g. Cholesky + Schur Complement
#     def update(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
#         L11 = self.K_fact
#         A21 = self.kernel.matrix(X_train, self.X_train)
#         A22 = self.kernel.matrix(X_train, X_train)
#         L21 = torch.linalg.solve(L11,A21.T).T
#         L22 = torch.linalg.cholesky(A22 - A21 @ torch.cholesky_solve(A21.T,L11))
#         L_left = torch.cat([L11, L21], dim=0)
#         L_right = torch.cat([torch.zeros_like(L21.T), L22], dim=0)
#         self.K_fact = torch.cat([L_left, L_right], dim=1)
#         self.X_train = torch.cat([self.X_train, X_train], dim=0)
#         self.y_train = torch.cat([self.y_train, y_train], dim=0)

#     def predict(self, X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         K_s = self.kernel.matrix(self.X_train, X_test)
#         K_ss = self.kernel.matrix(X_test, X_test)

#         KsT_K_inv = torch.cholesky_solve(K_s,self.K_fact).T
        
#         mu_s = KsT_K_inv @ self.y_train
#         cov_s = K_ss - KsT_K_inv @ K_s

#         return mu_s, cov_s