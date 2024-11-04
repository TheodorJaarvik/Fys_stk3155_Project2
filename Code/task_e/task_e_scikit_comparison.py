import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load the Wisconsin Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)  # Reshape to column vector

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add intercept term to X (bias)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function with L2 regularization
def compute_cost(X, y, weights, lambda_reg):
    m = y.shape[0]
    h = sigmoid(X @ weights)
    epsilon = 1e-15  # To prevent log(0)
    h = np.clip(h, epsilon, 1 - epsilon)
    cost = (
        -1 / m
    ) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    reg_term = (lambda_reg / (2 * m)) * np.sum(np.square(weights[1:]))
    return cost + reg_term

# Gradient computation with L2 regularization
def compute_gradient(X, y, weights, lambda_reg):
    m = y.shape[0]
    h = sigmoid(X @ weights)
    gradient = (1 / m) * (X.T @ (h - y))
    # Apply regularization to weights (exclude bias term)
    gradient[1:] += (lambda_reg / m) * weights[1:]
    return gradient

# Logistic Regression using SGD
def logistic_regression_sgd(X, y, learning_rate=0.001, n_epochs=500, lambda_reg=0.01, batch_size=32):
    m, n = X.shape
    weights = np.zeros((n, 1))
    losses = []
    test_accuracies = []

    for epoch in range(n_epochs):
        # Shuffle data at the start of each epoch
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, m, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Compute gradient and update weights
            gradient = compute_gradient(X_batch, y_batch, weights, lambda_reg)
            weights -= learning_rate * gradient

        # Compute loss for monitoring
        loss = compute_cost(X, y, weights, lambda_reg)
        losses.append(loss)

        # Predict on the test set and calculate accuracy
        y_test_pred = sigmoid(X_test @ weights) >= 0.5
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_accuracies.append(test_accuracy)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return weights, losses, test_accuracies

# Parameters
learning_rate = 0.001
lambda_reg = 0.01  # Regularization strength
n_epochs = 500

# Train logistic regression using SGD
weights, losses, test_accuracies_manual = logistic_regression_sgd(
    X_train, y_train, learning_rate=learning_rate, n_epochs=n_epochs, lambda_reg=lambda_reg, batch_size=32
)

# Compare with Scikit-Learn's Logistic Regression
clf = LogisticRegression(penalty='l2', C=1/lambda_reg, max_iter=n_epochs)
clf.fit(X_train[:, 1:], y_train.ravel())  # Exclude intercept term

# Store Scikit-Learn test accuracies as a function of epochs for comparison
test_accuracies_sklearn = []
for epoch in range(1, n_epochs + 1):
    clf.max_iter = epoch
    clf.fit(X_train[:, 1:], y_train.ravel())
    y_test_pred_sklearn = clf.predict(X_test[:, 1:])
    test_accuracy_sklearn = accuracy_score(y_test, y_test_pred_sklearn)
    test_accuracies_sklearn.append(test_accuracy_sklearn)

# Plot the comparison of test accuracies as a function of epochs
plt.figure(figsize=(10, 6))
plt.plot(range(n_epochs), test_accuracies_manual, label='Manual Logistic Regression')
plt.plot(range(n_epochs), test_accuracies_sklearn, label='Scikit-Learn Logistic Regression', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Comparison of Manual vs. Scikit-Learn Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()

# Print final test accuracies for comparison
print(f"Final Test Accuracy (Manual Logistic Regression): {test_accuracies_manual[-1]:.4f}")
print(f"Final Test Accuracy (Scikit-Learn Logistic Regression): {test_accuracies_sklearn[-1]:.4f}")

# FFNN Test Accuracy (example value for comparison)
ffnn_test_accuracy = 0.96
print(f"FFNN Test Accuracy: {ffnn_test_accuracy:.4f}")
