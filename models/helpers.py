import numpy as np

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    
    return ary