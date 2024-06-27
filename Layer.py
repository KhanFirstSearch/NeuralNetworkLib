import numpy as np
import Math

class Layer:
    def __init__(self, input_size, output_size, activation):
        #Create random weights and biases
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))
        self.activation = activation

        #Ensure user has chosen a valid activation function
        if activation == 'sigmoid':
            self.activation_func = lambda x: Math.sigmoid(x)
            self.activation_derivative = lambda x: Math.sigmoid(x, deriv=True)
        elif activation == 'relu':
            self.activation_func = lambda x: Math.relu(x)
            self.activation_derivative = lambda x: Math.relu(x, deriv=True)
        else:
            raise ValueError("Unsupported activation function")

    #Feeds input forward to return the result
    def feed_forward(self, input_data):
        self.input_data = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        self.a = self.activation_func(self.z)
        return self.a

    #Back propagation to change weights, biases for more accurate output.
    def backward(self, output_error, learning_rate):
        activation_error = output_error * self.activation_derivative(self.a)
        input_error = np.dot(activation_error, self.weights.T)
        weights_error = np.dot(self.input_data.T, activation_error)

        #Update parameters
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * np.sum(activation_error, axis=0, keepdims=True)
        return input_error
