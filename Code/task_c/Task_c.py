import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product

# Data generation function
def generate_data(n_samples=100):
    np.random.seed(0)
    x = np.random.rand(n_samples)
    y = 2 + 3*x + 4*x**2 + 0.1 * np.random.randn(n_samples)
    return x, y

# Generate data
x, y = generate_data()
X = np.c_[np.ones(x.shape), x, x**2]
y = y.reshape(-1, 1)

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, z * alpha)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

# Activation function dictionary
activation_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative),
    "linear": (lambda x: x, lambda x: np.ones_like(x))  # Linear for output layer
}

# Initialize network
def initialize_network(n_inputs, n_hidden_layers, nodes_per_layer, n_outputs, activations):
    np.random.seed(0)
    network = {}
    layer_sizes = [n_inputs] + nodes_per_layer + [n_outputs]

    for layer in range(n_hidden_layers + 1):
        network[f"W{layer}"] = np.random.randn(layer_sizes[layer], layer_sizes[layer + 1]) * 0.1
        network[f"b{layer}"] = np.zeros((1, layer_sizes[layer + 1]))
        network[f"activation{layer}"] = activations[layer]

    return network

# Loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Forward propagation
def forward_propagation(X, network):
    activations = {"A0": X}
    for layer in range(len(network) // 3):
        W = network[f"W{layer}"]
        b = network[f"b{layer}"]
        act_fn, _ = activation_functions[network[f"activation{layer}"]]

        Z = activations[f"A{layer}"] @ W + b
        activations[f"A{layer + 1}"] = act_fn(Z) if layer < len(network) // 3 - 1 else Z

    return activations

# Backward propagation
def back_propagation(y, activations, network):
    gradients = {}
    n_layers = len(network) // 3
    y_pred = activations[f"A{n_layers}"]

    dA = y_pred - y
    for layer in reversed(range(n_layers)):
        _, act_derivative = activation_functions[network[f"activation{layer}"]]
        dZ = dA if layer == n_layers - 1 else dA * act_derivative(activations[f"A{layer + 1}"])
        dW = activations[f"A{layer}"].T @ dZ / y.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / y.shape[0]

        gradients[f"dW{layer}"] = dW
        gradients[f"db{layer}"] = db
        if layer > 0:
            dA = dZ @ network[f"W{layer}"].T

    return gradients

# Update parameters
def update_parameters(network, gradients, learning_rate):
    for layer in range(len(network) // 3):
        network[f"W{layer}"] -= learning_rate * gradients[f"dW{layer}"]
        network[f"b{layer}"] -= learning_rate * gradients[f"db{layer}"]
    return network

# Training function with mini-batch SGD
def train_neural_network_sgd(X, y, network, n_epochs=1000, learning_rate=0.01, batch_size=16):
    n_samples = X.shape[0]
    losses = []

    for epoch in range(n_epochs):
        # Shuffle data at the start of each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass
            activations = forward_propagation(X_batch, network)
            y_pred = activations[f"A{len(network) // 3}"]

            # Compute loss for the batch
            batch_loss = mse_loss(y_batch, y_pred)
            epoch_loss += batch_loss * len(y_batch)  # Accumulate loss over all batches

            # Backward pass
            gradients = back_propagation(y_batch, activations, network)

            # Update parameters
            network = update_parameters(network, gradients, learning_rate)

        # Record average loss for the epoch
        epoch_loss /= n_samples
        losses.append(epoch_loss)


    return network, losses

# Model parameters
n_inputs = X.shape[1]
n_hidden_layers = 3
nodes_per_layer = [10, 10, 10]
n_outputs = 1
all_losses = []
optimal = [[], 100]
n_epochs = 100
learning_rate = 0.01

activations = ["sigmoid", "relu", "leaky_relu"]
combination_length = len(activations)
activation_permutations = list(product(activations, repeat=combination_length))
activation_permutations = [list(item) for item in activation_permutations]
for permutation in activation_permutations:
    permutation.append("linear")

for actv in activation_permutations:

    network = initialize_network(n_inputs, n_hidden_layers, nodes_per_layer, n_outputs, actv)
    network, sgd_losses = train_neural_network_sgd(X, y, network, n_epochs, learning_rate, batch_size=16  )
    current = [actv, sgd_losses[n_epochs-1]]
    print(f"Configuration: {current[0]}, MSE: {current[1]}")

    if current[1] < optimal[1]:
        optimal = current
    all_losses.append(sgd_losses)


for idx, ls in enumerate(all_losses):
    color = 'blue' if 'sigmoid' in activation_permutations[idx] else 'orange'
    plt.plot(ls, color=color, label=f"Config {idx}: {activation_permutations[idx]}")

plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title(f"Training Loss Over {n_epochs} Epochs for Different Configurations. L-rate = {learning_rate}")

plt.show()

print(f"Best performance: {optimal}")


