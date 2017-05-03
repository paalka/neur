import numpy as np

def mini_batch_partitioner(training_set, validation_set, batch_size=25):
    X_train = training_set[:][0]
    T_train = training_set[:][1]

    X_validation = validation_set[:][0]
    T_validation = validation_set[:][1]

    n_batches = X_train.shape[0] / batch_size
    XT_batches = zip(
        np.array_split(X_train, n_batches, axis=0),
        np.array_split(T_train, n_batches, axis=0))

    return XT_batches, X_validation, T_validation
