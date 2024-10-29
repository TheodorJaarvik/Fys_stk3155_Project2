import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Use the same generate_data function from Part a"  

def generate_data(n_samples=100):
    np.random.seed(0)
    x = np.random.rand(n_samples)
    y = 2 + 3*x + 4*x**2 + 0.1 * np.random.randn(n_samples)  
    return x, y

x, y = generate_data()
X = np.c_[np.ones(x.shape), x, x**2]
y = y.reshape(-1, 1)

def initialize_network(n_inputs, n_hidden_layers, nodes_per_layer, n_outputs):
    np.random.seed(0)
    network = {}

    for layer in range(n_hidden_layers + 1):
        if layer == 0: 
            input_size = n_inputs
        else: 
            input_size = nodes_per_layer
        output_size = nodes_per_layer if layer < n_hidden_layers else n_outputs
        

        network[f"W{layer}"] = np.random.randn(input_size, output_size) * 0.1
        network[f"b{layer}"] = np.zeros((1, output_size))  
    return network


n_inputs = X.shape[1]
n_hidden_layers = 2  # Flexible: Change number of hidden layers
nodes_per_layer = 10  # Flexible: Change number of nodes in hidden layers
n_outputs = 1
network = initialize_network(n_inputs, n_hidden_layers, nodes_per_layer, n_outputs)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def forward_propagation(X, network):
    activations = {"A0": X}
    for layer in range(len(network) // 2):
        W = network[f"W{layer}"]
        b = network[f"b{layer}"]
        Z = activations[f"A{layer}"] @ W + b
        if layer < len(network) // 2 - 1:
            activations[f"A{layer + 1}"] = sigmoid(Z) 
        else:
            activations[f"A{layer + 1}"] = Z  # Output layer (linear)
    return activations
def back_propagation(y, activations, network):
    gradients = {}
    n_layers = len(network) // 2
    y_pred = activations[f"A{n_layers}"]
    

    dA = y_pred - y
    for layer in reversed(range(n_layers)):
        dZ = dA if layer == n_layers - 1 else dA * sigmoid_derivative(activations[f"A{layer + 1}"])
        dW = activations[f"A{layer}"].T @ dZ / y.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / y.shape[0]
        gradients[f"dW{layer}"] = dW
        gradients[f"db{layer}"] = db
        if layer > 0:
            dA = dZ @ network[f"W{layer}"].T
    return gradients

def update_parameters(network, gradients, learning_rate):
    for layer in range(len(network) // 2):
        network[f"W{layer}"] -= learning_rate * gradients[f"dW{layer}"]
        network[f"b{layer}"] -= learning_rate * gradients[f"db{layer}"]
    return network

def train_neural_network(X, y, network, n_epochs=1000, learning_rate=0.01):
    losses = []
    for epoch in range(n_epochs):

        activations = forward_propagation(X, network)
        y_pred = activations[f"A{len(network) // 2}"]
        

        loss = mse_loss(y, y_pred)
        losses.append(loss)
        

        gradients = back_propagation(y, activations, network)
        

        network = update_parameters(network, gradients, learning_rate)
        

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return network, losses

# Train the model
network, losses = train_neural_network(X, y, network, n_epochs=1000, learning_rate=0.01)

# Plot training loss
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Epochs")
plt.show()

# Final prediction
final_activations = forward_propagation(X, network)
y_pred = final_activations[f"A{len(network) // 2}"]

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Final MSE: {mse:.4f}, R^2: {r2:.4f}")

