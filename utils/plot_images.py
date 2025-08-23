from matplotlib.pylab import imshow
import matplotlib.pyplot as plt
import numpy as np

class PlotImages:
    def display_data(self, X, y):
        fig, ax = plt.subplots(
            nrows=2,
            ncols=5,
            sharex=True,
            sharey=True
        )

        ax = ax.flatten()
        for i in range(10):
            img = X[y == i][0].reshape(28,28)
            ax[i].imshow(img, cmap='Greys')
        
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.savefig("plots/rescaled-nums.png")
        # plt.show() 

    def display_multiple_nums(self, X, y):
        fig, ax = plt.subplots(
            nrows=5,
            ncols=5,
            sharex=True,
            sharey=True
        )

        ax = ax.flatten()
        for i in range(25):
            img = X[y == 5][i].reshape(28,28)
            ax[i].imshow(img, cmap='Greys')
        
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.savefig("plots/5.png")
        # plt.show()

    def display_missclassified(self, model, X_test, y_test):
        X_test_subset = X_test[:1000, :]
        y_test_subset = y_test[:1000]

        _, probas = model.forward(X_test_subset)
        test_pred = np.argmax(probas, axis=1)

        miss_cl_img = X_test_subset[y_test_subset != test_pred][:25]
        miss_cl_labels = test_pred[y_test_subset != test_pred][:25]
        corr_labels = y_test_subset[y_test_subset != test_pred][:25]

        fig, ax = plt.subplots(
            nrows=5,
            ncols=5,
            sharex=True,
            sharey=True,
            figsize=(8,8)
        )

        ax = ax.flatten()

        for i in range(25):
            img = miss_cl_img[i].reshape(28,28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
            ax[i].set_title(
                f"{i+1}) "
                f"True: {corr_labels[i]}\n"
                f"Predicted: {miss_cl_labels[i]}"
            )
        
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.savefig("results/missclassified.png")