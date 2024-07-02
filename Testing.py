import NeuralNetwork as n
import Layer as l
import RNNLayer as rl
import numpy as np

#Neural Network
nn = n.NeuralNetwork()
nn.add_layer(l.Layer(2, 5, 'relu'))     #Input layer
nn.add_layer(l.Layer(5, 5, 'relu'))     #Hidden layer
nn.add_layer(l.Layer(5, 1, 'sigmoid'))  #Output layer

#Normalizing Function
#TODO: Find a way to not use this in testing area.
def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data - mean
    return (data - mean) / std

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

x_input = np.array([
    [185, 85],   #Should Predict a Male
    [130, 58],   #Should Predict a Female
])

predictions = nn.forward(x_input)
print("VNN Predictions:", predictions)

x_train2 = [
    normalize(np.array([[1], [2], [3], [4]])),
    normalize(np.array([[2], [3], [4], [5]])),
    normalize(np.array([[3], [4], [5], [6]])),
]

y_train2 = [
    normalize(np.array([[5]])),
    normalize(np.array([[6]])),
    normalize(np.array([[7]])),
]

rnn = n.NeuralNetwork()
rnn.add_layer(rl.RNNLayer(input_size=1, output_size=2, sequence_length=4))

for x_train_seq, y_train_seq in zip(x_train2, y_train2):
    rnn.train(x_train_seq, y_train_seq, epochs=1000, learning_rate=0.01)

x_input2 = np.array([[4], [5], [6], [7]])
prediction = rnn.forward(x_input2)
print("RNN Prediction:", prediction)