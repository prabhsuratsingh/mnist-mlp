from data.dataset import MNISTDataset as dataset
from data.preprocess import Preprocess
from models.mlp import MLP


def main():
    dataset.load()
    X, y = dataset.X, dataset.y

    X_train, y_train, X_valid, y_valid, X_test, y_test = Preprocess.split_data(X, y)

    model = MLP(
        num_features=28*28,
        num_hidden=50,
        num_classes=10
    )

if __name__ == "__main__":
    main()