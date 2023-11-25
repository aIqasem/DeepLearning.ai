import numpy as np


def gradient_descent_univariate(x, y, learning_rate, max_epochs):
    m = len(y)  # Number of training examples
    w = 0  # Initial value for the parameter w
    b = 0  # Initial value for the parameter b
    l_cost = np.array([])
    
    for i in range(max_epochs):
        # Calculate the predicted values
        y_pred = w * x + b

        # Calculate the gradient
        gradient_w = (1/m) * sum((y_pred - y) * x)
        gradient_b = (1/m) * sum((y_pred - y))
        
        # Update the parameter using the gradient and learning rate
        w = w - learning_rate * gradient_w
        b = b - learning_rate * gradient_b
            
        # Calculate and print the cost (mean squared error)
        cost = (1/(2 * m)) * sum((y_pred - y)**2)
        l_cost =  np.append(l_cost,[cost])
    
        if len(l_cost) > 1:
            if abs(l_cost[-1] - l_cost[-2]) <= 1e-10:
                print(f"last iteration: {i}")
                return w,b,cost
            
    return w,b,cost


def gradient_descent_multivariable(X, y, learning_rate = 0.01, bias=False):
    if bias:
        X = np.c_[np.ones((X.shape[0], 1)), X]
    
    m, n = X.shape
    y = y.reshape(m,1)
    theta = np.zeros((n, 1))  # Initialize parameters to zeros
    cost_history = []

    
    while len(cost_history)<100000:  # 100000 = maximum iterations
        # Calculate predictions
        predictions = np.dot(X, theta)
    
        # Calculate the error
        error = predictions - y
        
        # Calculate the gradient
        gradient = np.dot(X.T, error) / m
        
        # Update parameters
        theta = theta - learning_rate * gradient

        # Calculate the cost
        cost = np.sum(error ** 2) / (2 * m)
        cost_history.append(cost)

        if len(cost_history) > 1:
            if abs(cost_history[-2]-cost_history[-1]) < 1e-20:
                return theta, cost_history

    return theta, cost_history
