import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from itertools import product


def generate_data(n_samples=100):
    np.random.seed(0)
    x = np.random.rand(n_samples)
    y = 2 + 3 * x + 4 * x**2 + 0.1 * np.random.randn(n_samples)
    return x, y

x, y = generate_data()
X = np.c_[np.ones(x.shape), x, x**2]
y = y.reshape(-1, 1)

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

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Also tried initializing weights and biases using normal distribution, without seeing big differences in outputs
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

def forward_propagation(X, network, activation_functions):
    activations = {"A0": X}
    n_layers = len(network) // 2

    for layer in range(n_layers):
        W = network[f"W{layer}"]
        b = network[f"b{layer}"]
        Z = activations[f"A{layer}"] @ W + b


        if layer < n_layers - 1:
            if activation_functions[layer] == "sigmoid":
                activations[f"A{layer + 1}"] = sigmoid(Z)
            elif activation_functions[layer] == "relu":
                activations[f"A{layer + 1}"] = relu(Z)
            elif activation_functions[layer] == "leaky_relu":
                activations[f"A{layer + 1}"] = leaky_relu(Z)
        else:
            activations[f"A{layer + 1}"] = Z

    return activations

def back_propagation(y, activations, network, activation_functions):
    gradients = {}
    n_layers = len(network) // 2
    y_pred = activations[f"A{n_layers}"]

    dA = y_pred - y
    for layer in reversed(range(n_layers)):
        if layer == n_layers - 1:
            dZ = dA
        elif activation_functions[layer] == 'sigmoid':
            dZ = dA * sigmoid_derivative(activations[f"A{layer + 1}"])
        elif activation_functions[layer] == 'relu':
            dZ = dA * relu_derivative(activations[f"A{layer + 1}"])
        elif activation_functions[layer] == 'leaky_relu':
            dZ = dA * leaky_relu_derivative(activations[f"A{layer + 1}"])

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

def train_neural_network(X, y, network, activation_functions, n_epochs, learning_rate):
    losses = []
    for epoch in range(n_epochs):
        activations = forward_propagation(X, network, activation_functions)
        y_pred = activations[f"A{len(network) // 2}"]

        loss = mse_loss(y, y_pred)
        losses.append(loss)

        gradients = back_propagation(y, activations, network, activation_functions)
        network = update_parameters(network, gradients, learning_rate)

    return network, losses


activation_functions = ["sigmoid", "leaky_relu", "relu"]
combination_length = len(activation_functions)
activation_permutations = list(product(activation_functions, repeat=combination_length))
activation_permutations = [list(item) for item in activation_permutations]

n_inputs = X.shape[1]
n_hidden_layers = 3
nodes_per_layer = 10
n_outputs = 1
all_losses = []

# Training the network using each permutation of activation functions
# Seems that a learning rate of 0.01 works best to create a gradual fit
for actv in activation_permutations:
    network = initialize_network(n_inputs, n_hidden_layers, nodes_per_layer, n_outputs)
    network, losses = train_neural_network(X, y, network, actv, n_epochs=1000, learning_rate=0.01)

    all_losses.append(losses)


for idx, losses in enumerate(all_losses):
    plt.plot(losses, label=f"Config {idx + 1}: {activation_permutations[idx]}")

plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Epochs for Different Configurations")
plt.show()



for idx, actv in enumerate(activation_permutations):
    final_activations = forward_propagation(X, network, actv)
    y_pred = final_activations[f"A{len(network) // 2}"]


    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    if mse <= 1:
        print(f"Final MSE for configuration {idx + 1} {actv}: {mse:.4f}, R^2: {r2:.4f}")


