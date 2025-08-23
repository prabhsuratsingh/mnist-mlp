from sklearn.model_selection import train_test_split

class Preprocess:
    def split_data(self,X,y):
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=10000, random_state=123, stratify=y
        )

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp
        )

        return X_train, y_train, X_valid, y_valid, X_test, y_test