import Math

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.feed_forward(output)
        return output

    def backward(self, output_error, learning_rate):
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate)

    def train(self, x_train, y_train, epochs, learning_rate):
        #How many times do we want to train the model
        for epoch in range(epochs):
            #FeedForward input
            output = self.forward(x_train)

            #Compute loss (MSE)
            loss = Math.mse_loss(y_train, output)
            #Additional Info
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

            #Backward propagation
            output_error = Math.mse_loss(y_train, output, deriv=True)
            self.backward(output_error, learning_rate)
