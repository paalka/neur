import numpy as np
from sklearn.model_selection import train_test_split
from load_mnist_data import load_mnist_imgs, load_mnist_labels

from layers import LogisticLayer, SoftmaxOutputLayer, LinearLayer
from utils import get_accuracy, convert_to_one_hot
from NeuralNet import NeuralNet

hidden_layer_1_neurons = 100
hidden_layer_2_neurons = 100

layer_1 = (LinearLayer(28 * 28, hidden_layer_1_neurons), LogisticLayer())
layer_2 = (LinearLayer(hidden_layer_1_neurons, hidden_layer_2_neurons), LogisticLayer())
layer_3 = (LinearLayer(hidden_layer_2_neurons, 10), SoftmaxOutputLayer())

net = NeuralNet(layer_1, layer_2, layer_3)

training_mnist_imgs = load_mnist_imgs("data/train-images-idx3-ubyte") / 255.0
training_mnist_labels = convert_to_one_hot(load_mnist_labels("data/train-labels-idx1-ubyte"), 10)

test_mnist_imgs = load_mnist_imgs("data/t10k-images-idx3-ubyte") / 255.0
test_mnist_labels = convert_to_one_hot(load_mnist_labels("data/t10k-labels-idx1-ubyte"), 10)

X_train, X_validation, T_train, T_validation = train_test_split(training_mnist_imgs, training_mnist_labels, test_size=0.2)

learning_rate = 0.3
net.train((X_train, T_train), (X_validation, T_validation), learning_rate)

activations = net.feedforward(test_mnist_imgs)
true = np.argmax(test_mnist_labels, axis=1)
predictions = np.argmax(activations[-1], axis=1)

print(get_accuracy(predictions, true))
