import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Activation functions and their derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# Binary cross-entropy loss function
def binary_cross_entropy_loss(predictions, targets):
    epsilon = 1e-15  # To prevent log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

# Accuracy function
def compute_accuracy(predictions, targets):
    pred_labels = (predictions >= 0.5).astype(int)
    return np.mean(pred_labels == targets)

# Neural Network class
class NeuralNetwork:
    def __init__(self, layer_sizes, activations):
        self.layer_sizes = layer_sizes  # List of layer sizes [input_size, hidden1_size, ..., output_size]
        self.activations = activations  # List of activation functions per layer
        self.weights = []  # Weights for each layer
        self.biases = []   # Biases for each layer
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights and biases for each layer
        np.random.seed(0)  # For reproducibility
        for i in range(len(self.layer_sizes) - 1):
            input_dim = self.layer_sizes[i]
            output_dim = self.layer_sizes[i + 1]
            if self.activations[i] == 'sigmoid':
                # Xavier Initialization
                limit = np.sqrt(6 / (input_dim + output_dim))
                W = np.random.uniform(-limit, limit, (input_dim, output_dim))
            elif self.activations[i] == 'relu':
                # He Initialization
                W = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
            else:
                W = np.random.randn(input_dim, output_dim) * 0.01  # Default small weights
            b = np.zeros((1, output_dim))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        # Forward pass through the network
        self.z_values = []  # Linear combinations (Z) for each layer
        self.a_values = [X]  # Activations for each layer (A_0 = X)

        for i in range(len(self.weights)):
            Z = np.dot(self.a_values[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(Z)

            # Apply activation function
            if self.activations[i] == 'sigmoid':
                A = sigmoid(Z)
            elif self.activations[i] == 'relu':
                A = relu(Z)
            else:
                raise ValueError(f"Unsupported activation function: {self.activations[i]}")

            self.a_values.append(A)

        return self.a_values[-1]  # Return the final output layer (A_L)

    def backward(self, y, learning_rate):
        # Backward pass (compute gradients and update weights)
        m = y.shape[0]
        L = len(self.weights)
        dA = -(np.divide(y, self.a_values[-1]) - np.divide(1 - y, 1 - self.a_values[-1]))

        for i in reversed(range(L)):
            # Compute derivative of activation function
            if self.activations[i] == 'sigmoid':
                dZ = dA * sigmoid_derivative(self.a_values[i + 1])
            elif self.activations[i] == 'relu':
                dZ = dA * relu_derivative(self.z_values[i])
            else:
                raise ValueError(f"Unsupported activation function: {self.activations[i]}")

            dW = np.dot(self.a_values[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)

            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs, learning_rate):
        # Train the neural network using gradient descent
        losses = []
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)

            # Compute loss
            loss = binary_cross_entropy_loss(predictions, y)
            losses.append(loss)

            # Backward pass and update weights
            self.backward(y, learning_rate)

            # Optionally print loss and accuracy
            if epoch % 100 == 0 or epoch == epochs - 1:
                accuracy = compute_accuracy(predictions, y)
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

        return losses

    def predict(self, X):
        # Make predictions using the trained network
        predictions = self.forward(X)
        return (predictions >= 0.5).astype(int)



# Data Preparation
data = load_breast_cancer()
inputs = data.data
targets = data.target.reshape(-1, 1)  # Ensure targets are column vectors

# Standardize the inputs
scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)

# Network Architecture
input_size = inputs.shape[1]
output_size = 1
layer_sizes = [input_size, 16, 8, output_size]
activations = ['relu', 'relu', 'sigmoid']

# Create and Train the Neural Network
nn = NeuralNetwork(layer_sizes, activations)
epochs = 1000
learning_rate = 0.01
losses = nn.train(inputs, targets, epochs, learning_rate)

# Evaluate the Network
predictions = nn.forward(inputs)
accuracy = compute_accuracy(predictions, targets)
print(f'Final Accuracy on the dataset: {accuracy:.4f}')

# Plot Training Loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.title('Training Loss Over Time')
plt.show()
