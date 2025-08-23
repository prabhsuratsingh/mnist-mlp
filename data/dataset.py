from sklearn.datasets import fetch_openml

class MNISTDataset:
    X, y = None, None
    
    @classmethod
    def load(cls):

        X_data, y_data = fetch_openml("mnist_784", version=1, return_X_y=True)

        X_data = X_data.values
        y_data = y_data.astype(int).values

        # Normalizing pixel values to the range -1 to 1
        X_data = ((X_data / 255.) - .5) * 2

        cls.X = X_data
        cls.y = y_data

