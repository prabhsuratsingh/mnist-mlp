from models.helpers import onehot
import numpy as np

from training.batch import minibatch_generator

def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, corr_pred, num_examples = 0., 0, 0
    mg = minibatch_generator(X, y, minibatch_size)

    for i, (features, targets) in enumerate(mg):
        _, probas = nnet.forward(features)
        pred_labels = np.argmax(probas, axis=1)
        onehot_targets = onehot(targets, num_labels=num_labels)

        loss = np.mean((onehot_targets - probas)**2)
        corr_pred += (pred_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
    
    mse = mse/i
    acc = corr_pred/num_examples

    return mse, acc


def mse_loss(targets, probas, num_labels=10):
    onehot_targets = onehot(targets, num_labels=num_labels)
    
    return np.mean((onehot_targets - probas)**2)