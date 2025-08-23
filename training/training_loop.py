from training.batch import minibatch_generator
from training.loss import compute_mse_and_acc

minibatch_size = 100

def train(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        num_epochs,
        learning_rate=0.1
):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):
        mg = minibatch_generator(X_train, y_train, minibatch_size)

        for Xm, ym in mg:
            # compute outputs
            a_h, a_out = model.forward(Xm)

            # compute gradients
            d_loss__dw_out, \
            d_loss__db_out, \
            d_loss__d_w_h,  \
            d_loss__d_b_h = model.backward(Xm, a_h, a_out, ym)

            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__dw_out
            model.bias_out -= learning_rate * d_loss__db_out
        
        # logging 
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)

        train_acc, valid_acc = train_acc*100, valid_acc*100

        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)

        epoch_loss.append(train_mse)

        print(f"Epoch: {e+1:03d}/{num_epochs:03d} "
              f"| Train MSE: {train_mse:.2f}"
              f"| Train ACC: {train_acc:.2f}%"
              f"| Valid ACC: {valid_acc:.2f}%")
        
    return epoch_loss, epoch_train_acc, epoch_valid_acc