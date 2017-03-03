from __future__ import absolute_import
import numpy as np
from sklearn.model_selection import train_test_split
from examples.load_mnist_data import load_mnist_imgs, load_mnist_labels

from nn.layers import LinearLayer, TanhLayer, SoftmaxLayer
from nn.utils import get_accuracy, convert_to_one_hot
from nn.data_partitioners import mini_batch_partitioner
from nn.NeuralNet import NeuralNet

NORMALIZATION_CONSTANT = 255.0 # 255 is the highest value a pixel in the MNIST dataset can have.
MNIST_IMG_WIDTH = 28 # in pixels
MNIST_IMG_HEIGHT = 28 # in pixels

MNIST_N_POSSIBLE_VALUES = 10

training_mnist_imgs = load_mnist_imgs("data/train-images-idx3-ubyte") / NORMALIZATION_CONSTANT
training_mnist_labels = convert_to_one_hot(load_mnist_labels("data/train-labels-idx1-ubyte"), MNIST_N_POSSIBLE_VALUES)

test_mnist_imgs = load_mnist_imgs("data/t10k-images-idx3-ubyte") / NORMALIZATION_CONSTANT
test_mnist_labels = convert_to_one_hot(load_mnist_labels("data/t10k-labels-idx1-ubyte"), MNIST_N_POSSIBLE_VALUES)

hidden_layer_1_neurons = 100
hidden_layer_2_neurons = 100

layer_1 = (LinearLayer(MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT, hidden_layer_1_neurons), TanhLayer())
layer_2 = (LinearLayer(hidden_layer_1_neurons, hidden_layer_2_neurons), TanhLayer())
layer_3 = (LinearLayer(hidden_layer_2_neurons, MNIST_N_POSSIBLE_VALUES), SoftmaxLayer())

net = NeuralNet(mini_batch_partitioner, layer_1, layer_2, layer_3)

X_train, X_validation, T_train, T_validation = train_test_split(training_mnist_imgs, training_mnist_labels, test_size=0.2)

learning_rate = 0.3
costs = net.train((X_train, T_train), (X_validation, T_validation), learning_rate)

activations = net.feedforward(test_mnist_imgs)
true = np.argmax(test_mnist_labels, axis=1)
predictions = np.argmax(activations[-1], axis=1)

print(get_accuracy(predictions, true))
