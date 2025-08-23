from data.dataset import MNISTDataset as dataset
from data.preprocess import Preprocess
from models.mlp import MLP
from training.loss import compute_mse_and_acc
import numpy as np

from training.training_loop import train
from utils.metric_plots import plot_acc, plot_mse
from utils.plot_images import PlotImages


def main():
    dataset.load()
    X, y = dataset.X, dataset.y

    X_train, y_train, X_valid, y_valid, X_test, y_test = Preprocess.split_data(X, y)

    model = MLP(
        num_features=28*28,
        num_hidden=50,
        num_classes=10
    )

    np.random.seed(123)

    epoch_loss, epoch_train_acc, epoch_valid_acc = train(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        num_epochs=50,
        learning_rate=0.1
    )

    plot_mse(epoch_loss)
    plot_acc(epoch_train_acc, epoch_valid_acc)

    # test
    test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    plotter = PlotImages()
    plotter.display_missclassified(model, X_test, y_test)
    

if __name__ == "__main__":
    main()