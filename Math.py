import logging
import math

def sigmoid(x, deriv=False):
    if (deriv):
        return x * (1 - x)
    return 1 / (1 + math.exp(-x))

def dot(inputs, weights):
    if len(inputs) != len(weights):
        return -1

    sum = 0
    for i, num in enumerate(inputs):
        sum += inputs[i] * weights[i]
    return sum

def mse_loss(y_true, y_pred):
    if len(y_true) != len(y_pred):
        return -1
    sum = 0
    for i, num in enumerate(y_true):
        sum += (y_true[i] - y_pred[i]) ** 2
    return sum/len(y_true)