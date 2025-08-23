from data.dataset import MNISTDataset as dataset
from data.preprocess import Preprocess
from models.mlp import MLP
from training.loss import compute_mse_and_acc, mse_loss
from training.batch import minibatch_generator
import numpy as np

from utils.metrics import accuracy


def main():
    dataset.load()
    X, y = dataset.X, dataset.y

    X_train, y_train, X_valid, y_valid, X_test, y_test = Preprocess.split_data(X, y)

    model = MLP(
        num_features=28*28,
        num_hidden=50,
        num_classes=10
    )

    # for i in range(50):
    #     mg = minibatch_generator(X_train, y_train, 100)
    #     for Xm, ym in mg:
    #         break
    #     break

    # print(Xm.shape)
    # print(ym.shape)

    # _, probas = model.forward(X_valid)
    # mse = mse_loss(y_valid, probas)
    # print(f"Initial validation MSE: {mse:.1f}")

    # predicted_labels = np.argmax(probas, axis=1)
    # acc = accuracy(y_valid, predicted_labels)
    # print(f"Initial validation accuracy: {acc*100:.1f}")

    mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
    print(f"Initial validation MSE: {mse:.1f}")
    print(f"Initial validation accuracy: {acc*100:.1f}")

if __name__ == "__main__":
    main()