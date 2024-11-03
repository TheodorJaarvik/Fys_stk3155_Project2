import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the Wisconsin Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Activation function dictionary
activation_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative)
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
def binary_cross_entropy_loss(y_true, y_pred, network, lambda_reg):
    epsilon = 1e-15  # To prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    cross_entropy = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Compute L2 regularization term
    l2_term = 0
    num_layers = len(network) // 3
    for layer in range(num_layers):
        W = network[f"W{layer}"]
        l2_term += np.sum(np.square(W))
    l2_term = (lambda_reg / (2 * y_true.shape[0])) * l2_term
    
    return cross_entropy + l2_term

# Forward propagation
def forward_propagation(X, network):
    activations_cache = {"A0": X}
    pre_activations_cache = {}
    num_layers = len(network) // 3  # Number of layers
    
    for layer in range(num_layers):
        W = network[f"W{layer}"]
        b = network[f"b{layer}"]
        act_fn, _ = activation_functions[network[f"activation{layer}"]]
        
        Z = activations_cache[f"A{layer}"] @ W + b
        pre_activations_cache[f"Z{layer + 1}"] = Z
        activations_cache[f"A{layer + 1}"] = act_fn(Z)
    
    return activations_cache, pre_activations_cache

# Backward propagation with L2 regularization
def back_propagation(y, activations_cache, pre_activations_cache, network, lambda_reg):
    gradients = {}
    num_layers = len(network) // 3
    m = y.shape[0]
    y_pred = activations_cache[f"A{num_layers}"]
    
    # Compute initial gradient
    dZ = y_pred - y  # Derivative of BCE loss with sigmoid activation
    
    for layer in reversed(range(num_layers)):
        A_prev = activations_cache[f"A{layer}"]
        W = network[f"W{layer}"]
        act_fn, act_derivative = activation_functions[network[f"activation{layer}"]]
        
        dW = (A_prev.T @ dZ) / m + (lambda_reg / m) * W  # Include L2 regularization gradient
        db = np.sum(dZ, axis=0, keepdims=True) / m
        gradients[f"dW{layer}"] = dW
        gradients[f"db{layer}"] = db
        
        if layer > 0:
            Z_prev = pre_activations_cache[f"Z{layer}"]
            dA_prev = dZ @ W.T
            dZ = dA_prev * act_derivative(Z_prev)
    
    return gradients

# Update parameters
def update_parameters(network, gradients, learning_rate):
    num_layers = len(network) // 3
    for layer in range(num_layers):
        network[f"W{layer}"] -= learning_rate * gradients[f"dW{layer}"]
        network[f"b{layer}"] -= learning_rate * gradients[f"db{layer}"]
    return network

# Training function with L2 regularization
def train_neural_network_sgd(X, y, network, n_epochs=200, learning_rate=0.1, batch_size=32, lambda_reg=0.01):
    n_samples = X.shape[0]
    losses = []
    accuracies = []
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        epoch_accuracy = 0
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward pass
            activations_cache, pre_activations_cache = forward_propagation(X_batch, network)
            y_pred = activations_cache[f"A{len(network) // 3}"]
            
            # Compute loss and accuracy for the batch
            batch_loss = binary_cross_entropy_loss(y_batch, y_pred, network, lambda_reg)
            epoch_loss += batch_loss * len(y_batch)
            y_pred_labels = (y_pred >= 0.5).astype(int)
            batch_accuracy = accuracy_score(y_batch, y_pred_labels)
            epoch_accuracy += batch_accuracy * len(y_batch)
            
            # Backward pass
            gradients = back_propagation(y_batch, activations_cache, pre_activations_cache, network, lambda_reg)
            
            # Update parameters
            network = update_parameters(network, gradients, learning_rate)
        
        # Record average loss and accuracy for the epoch
        epoch_loss /= n_samples
        epoch_accuracy /= n_samples
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
    
    return network, losses, accuracies

# Train and evaluate helper function
def train_and_evaluate(n_hidden_layers, nodes_per_layer, activation, learning_rate, lambda_reg=0.01):
    network = initialize_network(X_train.shape[1], n_hidden_layers, nodes_per_layer, 1, [activation] * (n_hidden_layers + 1))
    network, _, _ = train_neural_network_sgd(X_train, y_train, network, n_epochs=100, learning_rate=learning_rate, batch_size=32, lambda_reg=lambda_reg)
    activations_cache, _ = forward_propagation(X_test, network)
    y_test_pred = activations_cache[f"A{len(network) // 3}"]
    y_test_pred_labels = (y_test_pred >= 0.5).astype(int)
    return accuracy_score(y_test, y_test_pred_labels)

# 1. Accuracy vs Learning Rate
learning_rates = [0.001, 0.01, 0.1, 0.5, 1, 10]
accuracy_lr = [train_and_evaluate(2, [16, 8], "sigmoid", lr) for lr in learning_rates]
plt.figure()
plt.plot(learning_rates, accuracy_lr, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Learning Rate")
plt.xscale("log")
plt.show()

# 2. Accuracy vs Activation Functions
activations = ["sigmoid", "tanh", "relu"]
accuracy_af = [train_and_evaluate(2, [16, 8], act, 0.1) for act in activations]
plt.figure()
plt.bar(activations, accuracy_af)
plt.xlabel("Activation Function")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Activation Function")
plt.show()

# 3. Accuracy vs Number of Nodes in Hidden Layers
node_configs = [[8, 8], [16, 8], [32, 16], [64, 32]]
accuracy_nodes = [train_and_evaluate(2, nodes, "sigmoid", 0.1) for nodes in node_configs]
plt.figure()
plt.plot([str(nodes) for nodes in node_configs], accuracy_nodes, marker='o')
plt.xlabel("Nodes per Layer")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Nodes per Layer")
plt.show()

# 4. Accuracy vs Number of Hidden Layers
layer_counts = [1, 2, 3, 4]
accuracy_layers = [train_and_evaluate(layers, [16] * layers, "sigmoid", 0.1) for layers in layer_counts]
plt.figure()
plt.plot(layer_counts, accuracy_layers, marker='o')
plt.xlabel("Number of Hidden Layers")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Hidden Layers")
plt.show()
