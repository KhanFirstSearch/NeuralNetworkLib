import numpy as np
import Math

class RNNLayer:
    def __init__(self, input_size, output_size, sequence_length):
        #Create random weights and biases
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.weights_input = np.random.randn(input_size, output_size) * 0.1
        self.weights_hidden = np.random.randn(output_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))
        self.hidden_state = np.zeros((1, output_size))

    def feed_forward(self, input_data):
        self.inputs = []         #A place to store inputs for backprop
        self.hidden_states = []  #A place to store hidden states for backprop

        #Loop through each step in the input seq
        for t in range(len(input_data)):
            input_t = input_data[t]
            self.inputs.append(input_t)

            #Compute the new hidden state using tanh activation function
            self.hidden_state = Math.tanh(
                np.dot(input_t, self.weights_input) + np.dot(self.hidden_state, self.weights_hidden) + self.biases)
            self.hidden_states.append(self.hidden_state)

        #Return the last hidden state as the output of the nn
        return self.hidden_state

    def backward(self, output_error, learning_rate):
        #Set the gradients for weights and biases with zeros
        d_weights_input = np.zeros_like(self.weights_input)
        d_weights_hidden = np.zeros_like(self.weights_hidden)
        d_biases = np.zeros_like(self.biases)
        d_hidden_state = np.zeros_like(self.hidden_state)

        for t in reversed(range(len(self.inputs))):
            input_t = self.inputs[t]
            hidden_state_t = self.hidden_states[t]
            d_output_error = output_error

            #Deriv of tanh func
            dtanh = d_output_error * (1 - hidden_state_t ** 2)
            d_biases += dtanh
            d_weights_input += np.dot(input_t.T, dtanh)
            d_weights_hidden += np.dot(self.hidden_states[t - 1].T, dtanh) if t > 0 else np.zeros_like(d_weights_hidden)

            #Backprop the error to previous step
            d_hidden_state = np.dot(dtanh, self.weights_hidden.T)

        #Update weights and biases
        self.weights_input -= learning_rate * d_weights_input
        self.weights_hidden -= learning_rate * d_weights_hidden
        self.biases -= learning_rate * d_biases

        #Return gradient for hidden state to prop through previous layers
        return d_hidden_state
