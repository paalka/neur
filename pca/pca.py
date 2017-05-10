import numpy as np
from sklearn import preprocessing

class PCA(preprocessing.FunctionTransformer):

    def __init__(self, n_components):
        super(PCA, self).__init__()
        self.n_components = n_components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        means = np.mean(all_samples, axis=1)

        cov_matrix = np.cov(all_samples)
        eig_val, eig_vec = np.linalg.eig(cov_matrix)

        eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        matrix_w = np.hstack([pair[1].reshape(pair[1].shape[0], 1) for pair in eig_pairs[:self.n_components]])
        return matrix_w.T.dot(all_samples - means[:,np.newaxis]).T
