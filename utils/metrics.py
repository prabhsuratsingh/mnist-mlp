import numpy as np

def accuracy(targets, pred_labels):
    return np.mean(pred_labels == targets)