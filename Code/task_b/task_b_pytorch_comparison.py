import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# Activation function dictionary
activation_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "linear": (lambda x: x, lambda x: np.ones_like(x))  # Linear for output layer
}

# Initialize network for manual implementation
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
network = initialize_network(n_inputs, n_hidden_layers, nodes_per_layer, n_outputs, activations)

# Loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Forward propagation for manual implementation
def forward_propagation(X, network):
    activations = {"A0": X}
    for layer in range(len(network) // 3):
        W = network[f"W{layer}"]
        b = network[f"b{layer}"]
        act_fn, _ = activation_functions[network[f"activation{layer}"]]
        
        Z = activations[f"A{layer}"] @ W + b
        activations[f"A{layer + 1}"] = act_fn(Z) if layer < len(network) // 3 - 1 else Z
    
    return activations

# Backward propagation for manual implementation
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

# Update parameters for manual implementation
def update_parameters(network, gradients, learning_rate):
    for layer in range(len(network) // 3):
        network[f"W{layer}"] -= learning_rate * gradients[f"dW{layer}"]
        network[f"b{layer}"] -= learning_rate * gradients[f"db{layer}"]
    return network

# Training function for manual implementation with mini-batch SGD
def train_neural_network_sgd(X, y, network, n_epochs=1000, learning_rate=0.01, batch_size=16):
    manual_losses = []
    n_samples = X.shape[0]
    
    for epoch in range(n_epochs):
        # Shuffle data at the beginning of each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward and backward pass
            activations = forward_propagation(X_batch, network)
            y_pred = activations[f"A{len(network) // 3}"]
            batch_loss = mse_loss(y_batch, y_pred)
            epoch_loss += batch_loss * len(y_batch)
            
            gradients = back_propagation(y_batch, activations, network)
            network = update_parameters(network, gradients, learning_rate)
        
        # Average loss per epoch
        epoch_loss /= n_samples
        manual_losses.append(epoch_loss)
        
        if epoch % 100 == 0:
            print(f"[Manual NN (SGD)] Epoch {epoch}, Loss: {epoch_loss:.4f}")
    
    return network, manual_losses

# Train the manual model with mini-batch SGD
network, manual_losses = train_neural_network_sgd(X, y, network, n_epochs=100, learning_rate=0.01, batch_size=16)

# PyTorch Implementation
# Convert data to PyTorch tensors
X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32)

# Define the neural network model in PyTorch
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_layers, nodes_per_layer, output_size, activations):
        super(SimpleNN, self).__init__()
        
        layers = []
        in_features = input_size
        
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_features, nodes_per_layer[i]))
            if activations[i] == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activations[i] == "relu":
                layers.append(nn.ReLU())
            in_features = nodes_per_layer[i]
        
        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Initialize PyTorch model
model = SimpleNN(n_inputs, n_hidden_layers, nodes_per_layer, n_outputs, activations)

# Loss function and optimizer for PyTorch
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop for PyTorch model with mini-batch SGD
pytorch_losses = []
for epoch in range(100):
    model.train()
    permutation = torch.randperm(X_torch.size(0))
    
    epoch_loss = 0
    for i in range(0, X_torch.size(0), 16):
        indices = permutation[i:i+16]
        X_batch, y_batch = X_torch[indices], y_torch[indices]
        
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        epoch_loss += loss.item() * len(y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    epoch_loss /= X_torch.size(0)
    pytorch_losses.append(epoch_loss)
    
    if epoch % 100 == 0:
        print(f"[PyTorch NN (SGD)] Epoch {epoch}, Loss: {epoch_loss:.4f}")

# Plot comparison of training loss
plt.plot(manual_losses, label="Manual NN (SGD) Loss")
plt.plot(pytorch_losses, label="PyTorch NN (SGD) Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss Comparison: Manual vs PyTorch (Mini-Batch SGD)")
plt.legend()
plt.show()
