import os
import argparse
import logging
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from jax import random
from minigp.nn import SimpleNet, SimpleNetParameters, SimpleNetState

os.makedirs("logs", exist_ok=True)
os.makedirs("results/simplenet", exist_ok=True)

logging.basicConfig(
    filename="logs/simplenet_loss_log.txt",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

parser = argparse.ArgumentParser(description="Train ICNN Model for f-fit and/or grad-fit.")
parser.add_argument("--cuda", action="store_true", help="Enable CUDA if available.")
parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
parser.add_argument("--steps", type=int, default=5000, help="Number of training steps.")
parser.add_argument("--func", type=str, default="quad", choices=["quad", "square"], help="Function to train on (quad=x^4, square=x^2).")
parser.add_argument("--mode", type=str, default="both", choices=["f-fit", "grad-fit", "both"], help="Train function fit, gradient fit, or both.")
parser.add_argument("--network-shape", type=str, default="1,32,32,1", help="Comma-separated layer dimensions of the ICNN.")
parser.add_argument("--loss-f", type=str, default="mse", choices=["mse", "mae"], help="Loss function for f-fit (Mean Squared Error or Mean Absolute Error).")
parser.add_argument("--loss-grad", type=str, default="mse", choices=["mse", "mae"], help="Loss function for grad-fit.")
args = parser.parse_args()
network_shape = list(map(int, args.network_shape.split(",")))


try:
    available_gpus = jax.devices("gpu")
except RuntimeError:
    available_gpus = []
if args.cuda and available_gpus:
    jax.config.update("jax_platform_name", "gpu")
    device = available_gpus[0]
    print("Using CUDA GPU:", device)
else:
    jax.config.update("jax_platform_name", "cpu")
    device = jax.devices("cpu")[0]
    print("No CUDA GPU detected. Running on CPU.")

def square_1d(x): return jnp.array([x[0]**2 - 1])
def square_1d_grad(x): return jnp.array([2 * x[0]])

def quad_1d(x): return jnp.array([x[0]**4 + 1])
def quad_1d_grad(x): return jnp.array([4 * x[0] ** 3])

funcs = {"quad": (quad_1d, quad_1d_grad), "square": (square_1d, square_1d_grad)}
func, grad = funcs[args.func]

def loss_f_fit(model_params, model_state, X, y):
    pred = SimpleNet.forward(X, model_state, model_params)
    pred = pred.reshape(-1, 1)
    if args.loss_f == "mse":
        print(pred.shape, y.shape)
        return jnp.mean((pred - y) ** 2)
    elif args.loss_f == "mae":
        return jnp.mean(jnp.abs(pred - y))

def loss_grad_fit(model_params, model_state, X, y_grad):
    grad_fn = jax.jit(jax.vmap(jax.grad(lambda x: SimpleNet.forward(x, model_state, model_params))))
    pred_grad = grad_fn(X)
    if args.loss_grad == "mse":
        print(pred_grad.shape, y_grad.shape)
        return jnp.mean((pred_grad - y_grad) ** 2)
    elif args.loss_grad == "mae":
        return jnp.mean(jnp.abs(pred_grad - y_grad))

def generate_synthetic_data(func, grad, n_samples=50, x_range=(-2.0, 2.0)):
    X = jnp.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
    Y = jax.vmap(func)(X)
    Ygrad = jax.vmap(grad)(X)
    return X, Y, Ygrad

def train_simplenet_f_fit(func, grad, lr, steps):
    X_train, Yf_train, _ = generate_synthetic_data(func, grad)
    X_train, Yf_train = jax.device_put(X_train, device), jax.device_put(Yf_train, device)

    key = random.PRNGKey(68)
    model_state = SimpleNetState(layer_dims=network_shape)
    model_params = SimpleNet.initialize_params(key, model_state)
    model_params = jax.device_put(model_params, device)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(model_params)

    @jax.jit
    def update(params, opt_state):
        loss_val, grads = jax.value_and_grad(loss_f_fit)(params, model_state, X_train, Yf_train)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_val

    logging.info(f"FUNCTION FITTING LOSS")
    for step in range(steps):
        model_params, opt_state, loss_val = update(model_params, opt_state)
        
        if step % 500 == 0:
            logging.info(f"[f-fit] Step {step}: Loss {loss_val:.6f}")

    return model_state, model_params, X_train, Yf_train

def train_simplenet_grad_fit(func, grad, lr, steps):
    X_train, _, Ygrad_train = generate_synthetic_data(func, grad)
    X_train, Ygrad_train = jax.device_put(X_train, device), jax.device_put(Ygrad_train, device)

    key = random.PRNGKey(68)
    model_state = SimpleNetState(layer_dims=network_shape)
    model_params = SimpleNet.initialize_params(key, model_state)
    model_params = jax.device_put(model_params, device)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(model_params)

    @jax.jit
    def update(params, opt_state):
        loss_val, grads = jax.value_and_grad(loss_grad_fit)(params, model_state, X_train, Yf_train)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_val

    logging.info(f"GRADIENT FITTING LOSS")
    for step in range(steps):
        model_params, opt_state, loss_val = update(model_params, opt_state)

        if step % 500 == 0:
            logging.info(f"[grad-fit] Step {step}: Loss {loss_val:.6f}")

    return model_state, model_params, X_train, Ygrad_train

def save_interpolated_graph(X_train, Y_true, Y_learned, step, mode, func_name):
    plt.figure(figsize=(8, 4))
    plt.plot(X_train, Y_true, 'k--', label="True")
    plt.plot(X_train, Y_learned, 'r-', label="Learned")
    plt.title(f"{mode} Approximation")
    plt.legend()

    filename = f"results/simplenet/{mode}_{func_name}_shape{args.network_shape.replace(',', '_')}_lossf_{args.loss_f}_lossg_{args.loss_grad}_step{step}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved graph: {filename}")

if __name__ == "__main__":
    if args.mode in ["f-fit", "both"]:
        model_state, model_params, X_train, Yf_train = train_simplenet_f_fit(func, grad, args.lr, args.steps)
        f_learned = SimpleNet.forward(X_train, model_state, model_params)
        save_interpolated_graph(X_train, Yf_train, f_learned, args.steps, "f-fit", args.func)

    if args.mode in ["grad-fit", "both"]:
        model_state, model_params, X_train, Ygrad_train = train_simplenet_grad_fit(func, grad, args.lr, args.steps)
        grad_learned = jax.jit(jax.vmap(jax.grad(lambda x: SimpleNet.forward(x, model_state, model_params))))(X_train)
        # save_interpolated_graph(X_train, Ygrad_train, grad_learned.squeeze(), args.steps, "grad-fit", args.func)
        save_interpolated_graph(X_train, Ygrad_train, grad_learned, args.steps, "grad-fit", args.func)