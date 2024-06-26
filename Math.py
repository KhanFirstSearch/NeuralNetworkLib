import numpy as np

def sigmoid(x, deriv=False):
    if (deriv):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def relu(x, deriv=False):
    if (deriv):
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)

def mse_loss(y_true, y_pred, deriv=False):
    if (deriv):
        return 2 * (y_pred - y_true) / y_true.size
    return np.mean((y_true - y_pred) ** 2)