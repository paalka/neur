import numpy as np
from sklearn.model_selection import train_test_split
from load_mnist_data import load_mnist_imgs, load_mnist_labels

from layers import TanhLayer, SoftmaxOutputLayer, LinearLayer
from utils import get_accuracy, convert_to_one_hot
from NeuralNet import NeuralNet

normalization_constant = 255.0 # 255 is the highest value a pixel in the MNIST dataset can have.
mnist_img_width = 28 # in pixels
mnist_img_height = 28 # in pixels

mnist_n_possible_values = 10

training_mnist_imgs = load_mnist_imgs("data/train-images-idx3-ubyte") / normalization_constant
training_mnist_labels = convert_to_one_hot(load_mnist_labels("data/train-labels-idx1-ubyte"), mnist_n_possible_values)

test_mnist_imgs = load_mnist_imgs("data/t10k-images-idx3-ubyte") / normalization_constant
test_mnist_labels = convert_to_one_hot(load_mnist_labels("data/t10k-labels-idx1-ubyte"), mnist_n_possible_values)

hidden_layer_1_neurons = 100
hidden_layer_2_neurons = 100

layer_1 = (LinearLayer(mnist_img_width * mnist_img_height, hidden_layer_1_neurons), TanhLayer())
layer_2 = (LinearLayer(hidden_layer_1_neurons, hidden_layer_2_neurons), TanhLayer())
layer_3 = (LinearLayer(hidden_layer_2_neurons, mnist_n_possible_values), SoftmaxOutputLayer())

net = NeuralNet(layer_1, layer_2, layer_3)

X_train, X_validation, T_train, T_validation = train_test_split(training_mnist_imgs, training_mnist_labels, test_size=0.2)

learning_rate = 0.3
net.train((X_train, T_train), (X_validation, T_validation), learning_rate)

activations = net.feedforward(test_mnist_imgs)
true = np.argmax(test_mnist_labels, axis=1)
predictions = np.argmax(activations[-1], axis=1)

print(get_accuracy(predictions, true))
