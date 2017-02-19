import numpy as np

def logistic(x):
    return 1 / (1 + np.exp(-x))

def dlogistic(x):
    return np.multiply(x, (1 - x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - x*x
