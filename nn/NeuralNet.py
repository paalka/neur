from utils.data_partitioners import mini_batch_partitioner
import autograd.numpy as np

class NeuralNet:

    def __init__(self, partitioner=mini_batch_partitioner, *layers):
        self.partitioner = partitioner

        self.layers = []
        self.cost = lambda Y, T: -np.sum(T * np.log(Y)) / Y.shape[0]
        for lin_projection, non_lin_trans in layers:
            self.layers.append(lin_projection)
            self.layers.append(non_lin_trans)

    def feedforward(self, X):
        for layer in self.layers:
            X = layer.get_output(X)

        return X

    def backpropagate(self, learning_rate, Y_predicted, correct_predictions):
        output_gradient = (Y_predicted - correct_predictions) / Y_predicted.shape[0]

        for layer in reversed(self.layers[:-1]):
            input_gradient = layer.get_input_gradient(output_gradient)

            layer.update_layer(output_gradient, learning_rate)

            output_gradient = input_gradient

    def train(self, training_set, validation_set, learning_rate=0.1, batch_size=32, n_iterations=30):
        XT_batches, X_validation, T_validation = self.partitioner(training_set, validation_set, batch_size)
        validation_costs = []

        for i in xrange(n_iterations):
            print("Started iteration: {} of {}".format(i+1, n_iterations))
            for X, T in XT_batches:
                Y_predicted = self.feedforward(X)
                self.backpropagate(learning_rate, Y_predicted, T)

            Y_predicted = self.feedforward(X_validation)
            validation_cost = self.cost(Y_predicted, T_validation)
            validation_costs.append(validation_cost)

            if len(validation_costs) > 3:
                if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
                    print("Cost did not decrease after three iterations! Quitting.")
                    return validation_costs

        return validation_costs
