import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_breast_cancer
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

def initialize_network(n_inputs, layers):
    np.random.seed(0)
    network = {}
    input_size = n_inputs

    i = 0

    for layer in layers:
        
        output_size = layer
        network[f"W{i}"] = np.random.randn(input_size, output_size) * 0.1
        network[f"b{i}"] = np.zeros((1, output_size))  

        input_size = output_size

        i += 1
    return network



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def ReLU(z):
    return np.maximum(0, z)

def RelU_derivative(z):
    return np.where(z > 0, 1, 0)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def forward_propagation(X, network, activation_funcs):
    activations = {"A0": X}
    for layer in range(len(network) // 2):
        W = network[f"W{layer}"]
        b = network[f"b{layer}"]
        Z = activations[f"A{layer}"] @ W + b
        if layer < len(network) // 2 - 1:
            activations[f"A{layer + 1}"] = activation_funcs[layer](Z) 
        else:
            activations[f"A{layer + 1}"] = Z  # Output layer (linear)
    return activations

def back_propagation(y, activations, network, a_func_derivatives):
    gradients = {}
    n_layers = len(network) // 2
    y_pred = activations[f"A{n_layers}"]
    

    dA = y_pred - y
    for layer in reversed(range(n_layers)):
        dZ = dA if layer == n_layers - 1 else dA * a_func_derivatives[layer](activations[f"A{layer + 1}"])
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

def train_neural_network(X, y, network, activation_funcs , a_func_derivatives, n_epochs=1000, learning_rate=0.01):
    losses = []
    for epoch in range(n_epochs):

        activations = forward_propagation(X, network, activation_funcs)
        y_pred = activations[f"A{len(network) // 2}"]
        

        loss = mse_loss(y, y_pred)
        losses.append(loss)
        

        gradients = back_propagation(y, activations, network, a_func_derivatives)
        

        network = update_parameters(network, gradients, learning_rate)
        

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return network, losses




n_inputs = X.shape[1]
n_outputs = 1



layers_sizes = [7, 3, n_outputs]
network = initialize_network(n_inputs, layers_sizes)



n_epochs = 1500
learning_rate = 0.01

activation_funcs = [sigmoid, sigmoid]
a_funcs_derivatives = [sigmoid_derivative, sigmoid_derivative]

# Train the model
network, losses = train_neural_network(X, y, network, activation_funcs, a_funcs_derivatives, n_epochs, learning_rate)

# Final prediction
final_activations = forward_propagation(X, network, activation_funcs)
y_pred = final_activations[f"A{len(network) // 2}"]

