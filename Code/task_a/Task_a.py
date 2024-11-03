import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def generate_data(n_samples=100):
    np.random.seed(0)
    x = np.random.rand(n_samples)
    y = 2 + 3*x + 4*x**2 + 0.1 * np.random.randn(n_samples)  
    return x, y

x, y = generate_data()
plt.scatter(x, y, label='Data points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data')
plt.legend()
plt.grid()
plt.show()

X = np.c_[np.ones(x.shape), x, x**2]  # For polynomial terms (1, x, x^2)
y = y.reshape(-1, 1) 

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    n_samples, n_features = X.shape
    theta = np.random.randn(n_features, 1)
    for i in range(n_iterations):
        gradients = (2 / n_samples) * X.T @ (X @ theta - y)
        theta -= learning_rate * gradients
    return theta

theta_gd = gradient_descent(X, y, learning_rate=0.1, n_iterations=1000)
print("Coefficients from GD:", theta_gd.ravel())

def gradient_descent_with_momentum(X, y, learning_rate=0.01, n_iterations=1000, momentum=0.9):
    n_samples, n_features = X.shape
    theta = np.random.randn(n_features, 1)
    velocity = np.zeros((n_features, 1))
    for i in range(n_iterations):
        gradients = (2 / n_samples) * X.T @ (X @ theta - y)
        velocity = momentum * velocity + learning_rate * gradients
        theta -= velocity
    return theta

theta_momentum = gradient_descent_with_momentum(X, y, learning_rate=0.1, n_iterations=1000)
print("Coefficients with Momentum:", theta_momentum.ravel())


def stochastic_gradient_descent(X, y, learning_rate=0.01, n_epochs=50, batch_size=20):
    n_samples, n_features = X.shape
    theta = np.random.randn(n_features, 1)
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, n_samples, batch_size):
            X_i = X_shuffled[i:i + batch_size]
            y_i = y_shuffled[i:i + batch_size]
            gradients = (2 / batch_size) * X_i.T @ (X_i @ theta - y_i)
            theta -= learning_rate * gradients
    return theta

theta_sgd = stochastic_gradient_descent(X, y, learning_rate=0.1, n_epochs=100, batch_size=20)
print("Coefficients from SGD:", theta_sgd.ravel())


def adagrad(X, y, learning_rate=0.1, n_iterations=1000, epsilon=1e-8):
    n_samples, n_features = X.shape
    theta = np.random.randn(n_features, 1)
    G = np.zeros((n_features, 1))
    for i in range(n_iterations):
        gradients = (2 / n_samples) * X.T @ (X @ theta - y)
        G += gradients**2
        theta -= (learning_rate / np.sqrt(G + epsilon)) * gradients
    return theta

theta_adagrad = adagrad(X, y, learning_rate=0.1, n_iterations=1000)
print("Coefficients from Adagrad:", theta_adagrad.ravel())


def rmsprop(X, y, learning_rate=0.01, n_iterations=1000, beta=0.9, epsilon=1e-8):
    n_samples, n_features = X.shape
    theta = np.random.randn(n_features, 1)
    G = np.zeros((n_features, 1))
    for i in range(n_iterations):
        gradients = (2 / n_samples) * X.T @ (X @ theta - y)
        G = beta * G + (1 - beta) * gradients**2
        theta -= (learning_rate / np.sqrt(G + epsilon)) * gradients
    return theta

theta_rmsprop = rmsprop(X, y, learning_rate=0.01, n_iterations=1000)
print("Coefficients from RMSprop:", theta_rmsprop.ravel())


def adam(X, y, learning_rate=0.01, n_iterations=1000, beta1=0.9, beta2=0.999, epsilon=1e-8):
    n_samples, n_features = X.shape
    theta = np.random.randn(n_features, 1)
    m = np.zeros((n_features, 1))
    v = np.zeros((n_features, 1))
    for i in range(1, n_iterations + 1):
        gradients = (2 / n_samples) * X.T @ (X @ theta - y)
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * gradients**2
        m_hat = m / (1 - beta1 ** i)
        v_hat = v / (1 - beta2 ** i)
        theta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return theta

# Run Adam
theta_adam = adam(X, y, learning_rate=0.01, n_iterations=1000)
print("Coefficients from Adam:", theta_adam.ravel())


# Helper function to calculate MSE and R^2
def evaluate_model(theta, X, y):
    predictions = X @ theta
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    return mse, r2

# Evaluate all models
methods = {
    "Gradient Descent": theta_gd,
    "Momentum": theta_momentum,
    "SGD": theta_sgd,
    "Adagrad": theta_adagrad,
    "RMSprop": theta_rmsprop,
    "Adam": theta_adam
}

for name, theta in methods.items():
    mse, r2 = evaluate_model(theta, X, y)
    print(f"{name} - MSE: {mse:.4f}, R^2: {r2:.4f}")


def plot_predictions(methods, X, y, x_vals):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_vals, y, color='black', label="Actual Data", alpha=0.6)
    

    for name, theta in methods.items():
        predictions = X @ theta
        plt.plot(x_vals, predictions, label=name)
    
    plt.xlabel("x")
    plt.ylabel("Predicted y")
    plt.title("Model Predictions vs Actual Data")
    plt.legend()
    plt.show()


x_vals_sorted = np.sort(x)
X_sorted = np.c_[np.ones(x_vals_sorted.shape), x_vals_sorted, x_vals_sorted**2]
plot_predictions(methods, X_sorted, y, x_vals_sorted)


