import NeuralNetwork as nn

class ExampleNeuralNetwork:
    def __init__(self, weights, bias):



        self.h1 = nn.Neuron(weights, bias)
        self.h2 = nn.Neuron(weights, bias)
        self.o1 = nn.Neuron(weights, bias)
        '''
        2 Neuron Input Layer
        2 Neuron Hidden Layer
        1 Neuron Output Layer
        '''

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        return self.o1.feedforward([out_h1, out_h2])
#-------------------
weights = [0, 1]
bias = 0
neuralNet = ExampleNeuralNetwork(weights, bias)

x = [2, 3]
print(neuralNet.feedforward(x))
#-------------------

y_true = ([1, 1, 0, 1])
y_pred = ([0, 0, 0, 0])

print(nn.mse_loss(y_true, y_pred))
