import numpy as np
from sklearn import preprocessing

class PCA(preprocessing.FunctionTransformer):

    def __init__(self, n_components):
        super(PCA, self).__init__()
        self.n_components = n_components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Use the eigenvectors of the covariance matrix to determine the
        # major directions of variation in the data.
        cov_matrix = np.cov(X)
        eig_val, eig_vec = np.linalg.eig(cov_matrix)

        eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        # Pick the 'n_components' vectors with the greatest variation.
        w = np.hstack([pair[1][:,np.newaxis] for pair in eig_pairs[:self.n_components]])

        sample_means = np.mean(X, axis=1)
        return w.T.dot(X - sample_means[:,np.newaxis]).T
