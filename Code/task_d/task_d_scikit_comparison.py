import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the Wisconsin Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Manual NN implementation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

activation_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative)
}

def initialize_network(n_inputs, n_hidden_layers, nodes_per_layer, n_outputs, activations):
    np.random.seed(0)
    network = {}
    layer_sizes = [n_inputs] + nodes_per_layer + [n_outputs]
    for layer in range(n_hidden_layers + 1):
        network[f"W{layer}"] = np.random.randn(layer_sizes[layer], layer_sizes[layer + 1]) * 0.1
        network[f"b{layer}"] = np.zeros((1, layer_sizes[layer + 1]))
        network[f"activation{layer}"] = activations[layer]
    return network

def binary_cross_entropy_loss(y_true, y_pred, network, lambda_reg):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    cross_entropy = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    l2_term = sum(np.sum(np.square(network[f"W{layer}"])) for layer in range(len(network) // 3))
    l2_term = (lambda_reg / (2 * y_true.shape[0])) * l2_term
    return cross_entropy + l2_term

def forward_propagation(X, network):
    activations_cache = {"A0": X}
    pre_activations_cache = {}
    num_layers = len(network) // 3
    for layer in range(num_layers):
        W, b = network[f"W{layer}"], network[f"b{layer}"]
        act_fn, _ = activation_functions[network[f"activation{layer}"]]
        Z = activations_cache[f"A{layer}"] @ W + b
        pre_activations_cache[f"Z{layer + 1}"] = Z
        activations_cache[f"A{layer + 1}"] = act_fn(Z)
    return activations_cache, pre_activations_cache

def back_propagation(y, activations_cache, pre_activations_cache, network, lambda_reg):
    gradients = {}
    num_layers = len(network) // 3
    m = y.shape[0]
    y_pred = activations_cache[f"A{num_layers}"]
    dZ = y_pred - y
    for layer in reversed(range(num_layers)):
        A_prev = activations_cache[f"A{layer}"]
        W, act_fn, act_derivative = network[f"W{layer}"], *activation_functions[network[f"activation{layer}"]]
        dW = (A_prev.T @ dZ) / m + (lambda_reg / m) * W
        db = np.sum(dZ, axis=0, keepdims=True) / m
        gradients[f"dW{layer}"] = dW
        gradients[f"db{layer}"] = db
        if layer > 0:
            Z_prev = pre_activations_cache[f"Z{layer}"]
            dA_prev = dZ @ W.T
            dZ = dA_prev * act_derivative(Z_prev)
    return gradients

def update_parameters(network, gradients, learning_rate):
    num_layers = len(network) // 3
    for layer in range(num_layers):
        network[f"W{layer}"] -= learning_rate * gradients[f"dW{layer}"]
        network[f"b{layer}"] -= learning_rate * gradients[f"db{layer}"]
    return network

def train_neural_network_sgd(X, y, network, n_epochs=200, learning_rate=0.1, batch_size=32, lambda_reg=0.01):
    n_samples = X.shape[0]
    accuracies = []
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled, y_shuffled = X[indices], y[indices]
        epoch_accuracy = 0
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]
            activations_cache, pre_activations_cache = forward_propagation(X_batch, network)
            y_pred = activations_cache[f"A{len(network) // 3}"]
            gradients = back_propagation(y_batch, activations_cache, pre_activations_cache, network, lambda_reg)
            network = update_parameters(network, gradients, learning_rate)
            y_pred_labels = (y_pred >= 0.5).astype(int)
            batch_accuracy = accuracy_score(y_batch, y_pred_labels)
            epoch_accuracy += batch_accuracy * len(y_batch)
        accuracies.append(epoch_accuracy / n_samples)
    return network, accuracies

# Initialize and train the manual neural network
n_inputs = X_train.shape[1]
n_hidden_layers = 2
nodes_per_layer = [16, 8]
network = initialize_network(n_inputs, n_hidden_layers, nodes_per_layer, 1, ["sigmoid", "sigmoid", "sigmoid"])
_, sgd_accuracies = train_neural_network_sgd(X_train, y_train, network, n_epochs=200, learning_rate=0.1)

# Scikit-learn Logistic Regression model
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train.ravel())

# Calculate Logistic Regression accuracy over epochs (simulated)
logistic_accuracies = []
for epoch in range(1, 201):
    y_train_pred = logistic_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    logistic_accuracies.append(train_accuracy)

# Plot comparison of manual NN and Logistic Regression accuracies over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(200), sgd_accuracies, label="Manual NN Accuracy")
plt.plot(range(200), logistic_accuracies, label="Logistic Regression Accuracy (Scikit-Learn)", linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Manual NN vs Logistic Regression - Accuracy over Epochs")
plt.legend()
plt.show()
