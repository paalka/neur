import numpy as np
from sklearn import base

class KNN(base.BaseEstimator):

    def __init__(self, k, distance_f=None):
        self.k = k
        if distance_f is None:
            self.distance_f = lambda u, v: np.linalg.norm(u-v, axis=1)
        else:
            self.distance_f = distance_f

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, y=None):
        n_test = X.shape[0]
        y_pred = np.zeros(n_test, dtype=self.y_train.dtype)

        for i in range(n_test):
            # Find the distance to each example in the training set
            distances = self.distance_f(self.X_train, X[i,:])

            k_closest_distances_idx = np.argsort(distances)[:self.k]
            k_closest_labels = np.take(self.y_train, k_closest_distances_idx)

            # Assume that the labels are always non-negative
            counts = np.bincount(k_closest_labels)
            y_pred[i] = np.argmax(counts)

        return y_pred
