from data.dataset import MNISTDataset as dataset
from data.preprocess import Preprocess


def main():
    dataset.load()
    X, y = dataset.X, dataset.y

    X_train, y_train, X_valid, y_valid, X_test, y_test = Preprocess.split_data(X, y)

if __name__ == "__main__":
    main()