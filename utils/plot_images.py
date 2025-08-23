import matplotlib.pyplot as plt

class PlotImages:
    def display_data(X, y):
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

    def display_multiple_nums(X, y):
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