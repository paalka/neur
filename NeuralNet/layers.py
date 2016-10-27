import numpy as np
from activation_functions import logistic, dlogistic, softmax, tanh, dtanh

class Layer:

    def get_params_gradient(self, X, output_gradient):
        return []

    def get_output(self, X, perform_dropout=False):
        pass

    def get_input_gradient(self, Y, output_gradient=None, T=None):
        pass


class LinearLayer(Layer):

    def __init__(self, n_in, n_out, do_dropout=False):
        self.W = np.random.normal(size=(n_in, n_out)) * 0.1
        self.b = np.zeros(n_out)
        self.do_dropout = do_dropout

    def get_output(self, X, perform_dropout=False):
        Y = X.dot(self.W) + self.b

        if perform_dropout and self.do_dropout:
            p = 0.7
            Y_d = (np.random.rand(*Y.shape) < p) / p
            Y *= Y_d

        return Y


    def get_input_gradient(self, Y, output_gradient):
        return output_gradient.dot(self.W.T)

    def get_weight_gradient(self, X, output_gradient):
        return X.T.dot(output_gradient)

    def get_bias_gradient(self, X, output_gradient):
        return np.sum(output_gradient, axis=0)


class LogisticLayer(Layer):

    def get_output(self, X, perform_dropout=False):
        return logistic(X)

    def get_input_gradient(self, Y, output_gradient):
        return dlogistic(Y) * output_gradient


class TanhLayer(Layer):

    def get_output(self, X, perform_dropout=False):
        return tanh(X)

    def get_input_gradient(self, Y, output_gradient):
        return dtanh(Y) * output_gradient


class SoftmaxOutputLayer(Layer):

    def get_output(self, X, perform_dropout=False):
        return softmax(X)

    def get_input_gradient(self, Y, T):
        return (Y - T) / Y.shape[0]

    def get_cost(self, Y, T):
        return -np.sum(T * np.log(Y)) / Y.shape[0]
