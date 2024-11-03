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

# Activation function dictionary
activation_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative)
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
n_inputs = X_train.shape[1]
n_hidden_layers = 2
nodes_per_layer = [16, 8]
n_outputs = 1
activations = ["sigmoid", "sigmoid", "sigmoid"]  # Including output layer activation
network = initialize_network(n_inputs, n_hidden_layers, nodes_per_layer, n_outputs, activations)

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
def train_neural_network_sgd(X, y, network, n_epochs=1000, learning_rate=0.01, batch_size=32, lambda_reg=0.01):
    n_samples = X.shape[0]
    losses = []
    accuracies = []
    
    for epoch in range(n_epochs):
        # Shuffle data at the start of each epoch
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
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    
    return network, losses, accuracies

# Train the model with L2 regularization
lambda_reg = 0.01  # Regularization strength
network, sgd_losses, sgd_accuracies = train_neural_network_sgd(
    X_train, y_train, network, n_epochs=200, learning_rate=0.1, batch_size=32, lambda_reg=lambda_reg
)

# Plot training loss and accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(sgd_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Training Loss Over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sgd_accuracies, label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Over Epochs")
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate on test data
activations_cache, _ = forward_propagation(X_test, network)
y_test_pred = activations_cache[f"A{len(network) // 3}"]
y_test_pred_labels = (y_test_pred >= 0.5).astype(int)
test_accuracy = accuracy_score(y_test, y_test_pred_labels)
print(f"Test Accuracy: {test_accuracy:.4f}")


