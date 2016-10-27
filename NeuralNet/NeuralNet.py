import numpy as np
from layers import LinearLayer

class NeuralNet:

    def __init__(self, *layers):
        self.layers = []
        for lin_projection, non_lin_trans in layers:
            self.layers.append(lin_projection)
            self.layers.append(non_lin_trans)

    def feedforward(self, input_data, perform_dropout=False):
        activations = [input_data]

        for layer in self.layers:
            X = activations[-1]
            Y = layer.get_output(X, perform_dropout)
            activations.append(Y)

        return activations

    def backpropagate(self, learning_rate, activations, correct_predictions):
        output_gradient = correct_predictions

        for layer in reversed(self.layers):
            Y = activations.pop()
            input_gradient = layer.get_input_gradient(Y, output_gradient)

            if isinstance(layer, LinearLayer):
                W_gradient = layer.get_weight_gradient(activations[-1], output_gradient)
                b_gradient = layer.get_bias_gradient(activations[-1], output_gradient)

                layer.W -= learning_rate * W_gradient
                layer.b -= learning_rate * b_gradient

            output_gradient = input_gradient

    def train(self, training_set, validation_set, learning_rate=0.1, batch_size=25, n_iterations=30):
        X_train = training_set[:][0]
        T_train = training_set[:][1]

        X_validation = validation_set[:][0]
        T_validation = validation_set[:][1]

        n_batches = X_train.shape[0] / batch_size
        XT_batches = zip(
            np.array_split(X_train, n_batches, axis=0),
            np.array_split(T_train, n_batches, axis=0))

        validation_costs = []
        for i in xrange(n_iterations):
            print("Started iteration: {} of {}".format(i+1, n_iterations))
            for X, T in XT_batches:
                activations = self.feedforward(X, perform_dropout=True)
                self.backpropagate(learning_rate, activations, T)

            activations = self.feedforward(X_validation)
            validation_cost = self.layers[-1].get_cost(activations[-1], T_validation)
            validation_costs.append(validation_cost)

            if len(validation_costs) > 3:
                if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
                    print("Cost did not decrease after three iterations! Quitting.")
                    return
