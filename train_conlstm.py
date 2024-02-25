import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from model.convLstm import ConvLSTM
from train_model import calc_error


def train_covlstm(
    model: torch.nn.Module,
    data: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.MSELoss,
    epochs=500,
):
    errors = []
    for t in range(epochs):
        # Process each mini-batch in turn:
        for x, y_actual in data:

            y_pred, _ = model(x)
            # Compute and print loss
            loss = criterion(
                y_pred.to(dtype=torch.float), y_actual[:, -1, :].to(dtype=torch.float)
            )
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Epoch ending, so now fit the coefficients based on all data:
        x, y_actual = data.dataset.tensors
        y_pred, _ = model(x)
        mse, rmse, perc_loss = calc_error(y_pred, y_actual[:, -1, :])
        errors.append(perc_loss)
        # Print some progress information as the net is trained:
        if epochs < 30 or t % 10 == 0:
            print(
                "epoch {:4d}: MSE={:.5f}, RMSE={:.5f} ={:.2f}%".format(
                    t, mse, rmse, perc_loss
                )
            )
