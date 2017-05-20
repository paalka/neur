from nn.optimizers import SGD
from sklearn import base
import autograd.numpy as np

class NeuralNet(base.BaseEstimator):

    def __init__(self, layers, optimizer=SGD, loss=None, cost=None):
        self.layers = layers
        self.optimizer = optimizer(self)

        if loss == None:
            self.loss = lambda Y, Y_predicted: (Y_predicted - Y) / Y_predicted.shape[0]
        else:
            self.loss = loss

        if cost == None:
            self.cost = lambda Y, Y_predicted: -np.sum(Y * np.log(Y_predicted)) / Y_predicted.shape[0]
        else:
            self.cost = cost

    def feedforward(self, X):
        for layer in self.layers:
            X = layer.get_output(X)

        return X

    def backpropagate(self, learning_rate, Y_predicted, Y):
        output_gradient = self.loss(Y, Y_predicted)

        for layer in reversed(self.layers[:-1]):
            input_gradient = layer.get_input_gradient(output_gradient)

            layer.update_layer(output_gradient, learning_rate)

            output_gradient = input_gradient

    def train(self, X, y, learning_rate=0.3, batch_size=32, n_iterations=30):
        validation_costs = self.optimizer(X, y)
        return validation_costs

    def fit(self, X, y=None):
        self.train(X, y)

    def predict(self, X, y=None):
        return self.feedforward(X)
