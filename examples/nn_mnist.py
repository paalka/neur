import autograd.numpy as np

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from nn.layers import LinearLayer, Activation
from nn.NeuralNet import NeuralNet
from nn.activation_functions import tanh, softmax
from utils.data_partitioners import mini_batch_partitioner
from utils.load_mnist_data import load_mnist_imgs, load_mnist_labels

NORMALIZATION_CONSTANT = 255.0 # 255 is the highest value a pixel in the MNIST dataset can have.
MNIST_IMG_WIDTH = 28 # in pixels
MNIST_IMG_HEIGHT = 28 # in pixels
MNIST_N_POSSIBLE_VALUES = 10

hidden_layer_1_neurons = 100
hidden_layer_2_neurons = 100

net = NeuralNet([LinearLayer(MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT, hidden_layer_1_neurons),
                 Activation(tanh),
                 LinearLayer(hidden_layer_1_neurons, hidden_layer_2_neurons),
                 Activation(tanh),
                 LinearLayer(hidden_layer_2_neurons, MNIST_N_POSSIBLE_VALUES),
                 Activation(softmax)
                ]
               )

training_mnist_imgs = (load_mnist_imgs("data/train-images-idx3-ubyte") / NORMALIZATION_CONSTANT).astype(np.float16)
one_hot_encoder = OneHotEncoder()
training_mnist_labels = one_hot_encoder.fit_transform(load_mnist_labels("data/train-labels-idx1-ubyte")).toarray()

pipeline = Pipeline([
                    ("nn_clf", net)
                ])


pipeline.fit(training_mnist_imgs, training_mnist_labels)
test_mnist_imgs = (load_mnist_imgs("data/t10k-images-idx3-ubyte") / NORMALIZATION_CONSTANT).astype(np.float16)
Y_predicted = pipeline.predict(test_mnist_imgs)

test_mnist_labels = one_hot_encoder.fit_transform(load_mnist_labels("data/t10k-labels-idx1-ubyte")).toarray()
true = np.argmax(test_mnist_labels, axis=1)
predictions = np.argmax(Y_predicted, axis=1)

print(accuracy_score(predictions, true))
