import numpy as np
from activation_functions import logistic, dlogistic, softmax, tanh, dtanh

class Layer:

    def get_params_gradient(self, X, output_gradient):
        return []

    def get_output(self, X):
        pass

    def get_input_gradient(self, Y, output_gradient=None, T=None):
        pass


class LinearLayer(Layer):

    def __init__(self, n_in, n_out):
        self.W = np.random.normal(size=(n_in, n_out)) * 0.1
        self.b = np.zeros(n_out)

    def get_output(self, X):
        return X.dot(self.W) + self.b

    def get_input_gradient(self, Y, output_gradient):
        return output_gradient.dot(self.W.T)

    def get_weight_gradient(self, X, output_gradient):
        return X.T.dot(output_gradient)

    def get_bias_gradient(self, X, output_gradient):
        return np.sum(output_gradient, axis=0)


class LogisticLayer(Layer):

    def get_output(self, X):
        return logistic(X)

    def get_input_gradient(self, Y, output_gradient):
        return dlogistic(Y) * output_gradient


class TanhLayer(Layer):

    def get_output(self, X):
        return tanh(X)

    def get_input_gradient(self, Y, output_gradient):
        return dtanh(Y) * output_gradient


class SoftmaxOutputLayer(Layer):

    def get_output(self, X):
        return softmax(X)

    def get_input_gradient(self, Y, T):
        return (Y - T) / Y.shape[0]

    def get_cost(self, Y, T):
        return -np.sum(T * np.log(Y)) / Y.shape[0]
