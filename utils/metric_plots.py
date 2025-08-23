import matplotlib.pyplot as plt

def plot_mse(epoch_loss):
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.savefig('results/mse_plot.png')

def plot_acc(epoch_train_acc, epoch_valid_acc):
    plt.plot(range(len(epoch_train_acc)), epoch_train_acc, label='Training')
    plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc, label='Validation')
    plt.ylabel('Accuracy')
    plt.xlabel("Epochs")
    plt.legend(loc='lower right')
    plt.savefig('results/acc_plot.png')