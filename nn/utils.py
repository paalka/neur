import numpy as np

def get_accuracy(predictions, true):
    num_correct = 0
    for i, label in enumerate(predictions):
        if label == true[i]:
            num_correct += 1

    return float(num_correct) / float(len(true))


def convert_to_one_hot(input_value, size):
    one_hot = np.zeros((len(input_value), size))
    one_hot[np.arange(len(one_hot)), input_value] += 1
    return one_hot

