import numpy as np

def euclidean(u, v):
    return np.linalg.norm(u-v, axis=1)
