import NeuralNetwork as n
import Layer as la
import numpy as np

#Neural Network
nn = n.NeuralNetwork()
nn.add_layer(la.Layer(2, 5, 'relu')) #Input layer
nn.add_layer(la.Layer(5, 5, 'relu')) #Hidden layer
nn.add_layer(la.Layer(5, 1, 'sigmoid')) #Output layer

#Height (cm), #Weight (kg)
x_train = np.array([
    [170, 70], #Male
    [160, 60], #Female
    [175, 80], #Male
    [155, 55], #Female
    [180, 90], #Male
    [165, 65], #Female
])

#1 = Male, #0 = Female
y_train = np.array([
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
])

#Train the neural network
nn.train(x_train, y_train, epochs=1000, learning_rate=0.1)

x_test = np.array([
    [185, 85], #Should Predict a Male
    [130, 58], #Should Predict a Female
])

predictions = nn.forward(x_test)
print("Predictions:", predictions)