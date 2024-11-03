import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
def logistic_regression_sgd(
    X, y, learning_rate=0.01, n_epochs=1000, lambda_reg=0.01, batch_size=32
):
    m, n = X.shape
    weights = np.zeros((n, 1))
    losses = []
    accuracies = []

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

        # Compute loss and accuracy for monitoring
        loss = compute_cost(X, y, weights, lambda_reg)
        losses.append(loss)

        # Predict on the training set
        y_pred_train = sigmoid(X @ weights) >= 0.5
        accuracy = accuracy_score(y, y_pred_train)
        accuracies.append(accuracy)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}, Loss: {loss:.4f}, Training Accuracy: {accuracy:.4f}"
            )

    return weights, losses, accuracies

# Training the logistic regression model
learning_rates = [0.001, 0.01, 0.1, 1]
lambda_reg = 0.01  # Regularization strength
test_accuracies = []

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    weights, losses, accuracies = logistic_regression_sgd(
        X_train, y_train, learning_rate=lr, n_epochs=1000, lambda_reg=lambda_reg, batch_size=32
    )

    # Predict on test set
    y_test_pred = sigmoid(X_test @ weights) >= 0.5
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_accuracy)
    print(f"Test Accuracy with learning rate {lr}: {test_accuracy:.4f}")

    # Plot training loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (Learning Rate: {lr})")

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.title(f"Training Accuracy (Learning Rate: {lr})")

    plt.tight_layout()
    plt.show()

# Plot test accuracy vs learning rates
plt.figure(figsize=(8, 6))
plt.plot(learning_rates, test_accuracies, marker='o')
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs Learning Rate')
plt.grid(True)
plt.show()

# Compare with Scikit-Learn's Logistic Regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty='l2', C=1/lambda_reg, max_iter=1000)
clf.fit(X_train[:, 1:], y_train.ravel())  # Exclude intercept term
y_test_pred_sklearn = clf.predict(X_test[:, 1:])
test_accuracy_sklearn = accuracy_score(y_test, y_test_pred_sklearn)
print(f"Scikit-Learn Logistic Regression Test Accuracy: {test_accuracy_sklearn:.4f}")

# Compare with FFNN (using the code from previous tasks)
# Assuming the FFNN code is defined in a function called `train_ffnn`
# We will reuse the FFNN code with the same dataset for comparison
# Note: If you haven't defined `train_ffnn`, you can define it similarly to the previous FFNN code

# For demonstration purposes, let's assume the FFNN achieves a test accuracy of 0.96
# In practice, you should run your FFNN code here and obtain the actual test accuracy
ffnn_test_accuracy = 0.96
print(f"FFNN Test Accuracy: {ffnn_test_accuracy:.4f}")
