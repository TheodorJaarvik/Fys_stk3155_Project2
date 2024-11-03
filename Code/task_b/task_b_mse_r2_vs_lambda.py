import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

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
activation_functions = {
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z: (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))),
    "linear": (lambda x: x, lambda x: np.ones_like(x))
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

# Model parameters
n_inputs = X.shape[1]
n_hidden_layers = 3
nodes_per_layer = [10, 5, 2]
n_outputs = 1
activations = ["sigmoid", "sigmoid", "sigmoid", "linear"]

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

# Backward propagation with L2 regularization
def back_propagation(y, activations, network, lambda_reg):
    gradients = {}
    n_layers = len(network) // 3
    y_pred = activations[f"A{n_layers}"]
    
    dA = y_pred - y
    for layer in reversed(range(n_layers)):
        _, act_derivative = activation_functions[network[f"activation{layer}"]]
        dZ = dA if layer == n_layers - 1 else dA * act_derivative(activations[f"A{layer + 1}"])
        dW = activations[f"A{layer}"].T @ dZ / y.shape[0] + (lambda_reg / y.shape[0]) * network[f"W{layer}"]
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
def train_neural_network_sgd(X, y, network, n_epochs=1000, learning_rate=0.01, batch_size=16, lambda_reg=0):
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
            epoch_loss += batch_loss * len(y_batch)
            
            # Backward pass
            gradients = back_propagation(y_batch, activations, network, lambda_reg)
            
            # Update parameters
            network = update_parameters(network, gradients, learning_rate)
        
        # Record average loss for the epoch
        epoch_loss /= n_samples
        losses.append(epoch_loss)
    
    return network, losses

# Lists to store results for MSE and R2 with different lambdas and learning rates
lambda_values = [0, 0.001, 0.01, 0.1, 1]
learning_rates = [0.001, 0.01, 0.1, 1]
mse_results_lambda = []
r2_results_lambda = []
mse_results_lr = []
r2_results_lr = []

# Test different lambda values
for lambda_reg in lambda_values:
    network = initialize_network(n_inputs, n_hidden_layers, nodes_per_layer, n_outputs, activations)
    network, _ = train_neural_network_sgd(X, y, network, n_epochs=500, learning_rate=0.01, batch_size=16, lambda_reg=lambda_reg)
    y_pred = forward_propagation(X, network)[f"A{len(network) // 3}"]
    mse_results_lambda.append(mean_squared_error(y, y_pred))
    r2_results_lambda.append(r2_score(y, y_pred))

# Plot MSE vs Lambda
plt.figure()
plt.plot(lambda_values, mse_results_lambda, marker='o')
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.title("MSE vs Lambda")
plt.xscale("log")
plt.show()

# Plot R2 vs Lambda
plt.figure()
plt.plot(lambda_values, r2_results_lambda, marker='o')
plt.xlabel("Lambda")
plt.ylabel("R2 Score")
plt.title("R2 vs Lambda")
plt.xscale("log")
plt.show()

# Test different learning rates
for lr in learning_rates:
    network = initialize_network(n_inputs, n_hidden_layers, nodes_per_layer, n_outputs, activations)
    network, _ = train_neural_network_sgd(X, y, network, n_epochs=500, learning_rate=lr, batch_size=16, lambda_reg=0.01)
    y_pred = forward_propagation(X, network)[f"A{len(network) // 3}"]
    mse_results_lr.append(mean_squared_error(y, y_pred))
    r2_results_lr.append(r2_score(y, y_pred))

# Plot MSE vs Learning Rate
plt.figure()
plt.plot(learning_rates, mse_results_lr, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("MSE")
plt.title("MSE vs Learning Rate")
plt.xscale("log")
plt.show()

# Plot R2 vs Learning Rate
plt.figure()
plt.plot(learning_rates, r2_results_lr, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("R2 Score")
plt.title("R2 vs Learning Rate")
plt.xscale("log")
plt.show()
