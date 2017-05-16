import autograd.numpy as np
from autograd import elementwise_grad


class Layer:

    def get_output(self, X):
        raise NotImplementedError()

    def get_input_gradient(self, output_gradient=None, T=None):
        raise NotImplementedError()

    def update_layer(self, output_gradient, learning_rate):
        raise NotImplementedError()


class LinearLayer(Layer):

    def __init__(self, n_in, n_out):
        self.W = np.random.normal(size=(n_in, n_out)) * 0.1
        self.b = np.zeros(n_out)

        self.prev_input = None

    def get_output(self, X):
        self.prev_input = X
        Y = X.dot(self.W) + self.b

        return Y

    def update_layer(self, output_gradient, learning_rate):
        W_gradient = self.get_weight_gradient(self.prev_input, output_gradient)
        b_gradient = self.get_bias_gradient(output_gradient)

        self.W -= learning_rate * W_gradient
        self.b -= learning_rate * b_gradient

    def get_input_gradient(self, output_gradient):
        return output_gradient.dot(self.W.T)

    def get_weight_gradient(self, X, output_gradient):
        return X.T.dot(output_gradient)

    def get_bias_gradient(self, output_gradient):
        return np.sum(output_gradient, axis=0)


class Activation(Layer):

    def __init__(self, activation_function):
        self.activation_function = activation_function
        self.activation_function_d = elementwise_grad(activation_function)
        self.prev_input = None

    def get_output(self, X):
        new_X = self.activation_function(X)
        self.prev_input = X
        return new_X

    def get_input_gradient(self, output_gradient):
        return self.activation_function_d(self.prev_input) * output_gradient

    def update_layer(self, output_gradient, learning_rate):
        pass
