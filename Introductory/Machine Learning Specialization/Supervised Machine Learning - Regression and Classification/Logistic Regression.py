import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Implementation of multivariate gradient descent for logistic regression 
def logistic_gradient_descent(X, y, learning_rate, max_iterrations=1000000):
    m, n = X.shape  # Number of training examples and features
    y = y.reshape(m,1)
    X = np.c_[np.ones((m, 1)), X]  # Add a column of ones for the bias term
    weights = np.zeros((n + 1, 1))  # Initialize weights with zeros, including the bias term
    cost_history = []

    for i in range(max_iterrations):
        # Calculate the predicted values
        z = np.dot(X, weights)
        y_pred = 1 / (1 + np.exp(-z))

        # Calculate the gradient
        gradient = (1/m) * np.dot(X.T, y_pred - y)

        # Update the weights
        weights -= learning_rate * gradient

        # Calculate and store the cost (log loss)
        cost = (-1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        cost_history.append(cost)

        if len(cost_history) > 1:
            if abs(cost_history[-1] - cost_history[-2]) <= 1e-7:
                print(f"Last iteration: {i}")
                return weights, cost_history

    return weights, cost_history