from utils.data_partitioners import mini_batch_partitioner
from sklearn.model_selection import train_test_split
from sklearn import base
import autograd.numpy as np

class NeuralNet(base.BaseEstimator):

    def __init__(self, layers, loss=None, cost=None, partitioner=mini_batch_partitioner):
        self.partitioner = partitioner

        self.layers = layers
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

    def train(self, X, y, learning_rate=0.1, batch_size=32, n_iterations=30):
        X_train, X_validation, T_train, T_validation = train_test_split(X, y, test_size=0.2)
        training_set = (X_train, T_train)
        validation_set = (X_validation, T_validation)
        XT_batches, X_validation, T_validation = self.partitioner(training_set, validation_set, batch_size)
        validation_costs = []

        for i in xrange(n_iterations):
            print("Started iteration: {} of {}".format(i+1, n_iterations))
            for X, Y in XT_batches:
                Y_predicted = self.feedforward(X)
                self.backpropagate(learning_rate, Y_predicted, Y)

            Y_predicted = self.feedforward(X_validation)
            validation_cost = self.cost(T_validation, Y_predicted)
            validation_costs.append(validation_cost)

            if len(validation_costs) > 3:
                if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
                    print("Cost did not decrease after three iterations! Quitting.")
                    return validation_costs

        return validation_costs

    def fit(self, X, y=None):
        self.train(X, y)

    def predict(self, X, y=None):
        return self.feedforward(X)
