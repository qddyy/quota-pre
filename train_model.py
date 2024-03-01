import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


from data.lstm_datloader import lstm_train_data
from model.vgg_lstm import VGG_LSTM


batch_size = 64
input_dim = 20
hidden_dim = 100
seq_len = 50
num_layers = 1
class_num = 5
batch_first = True


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.distance = torch.tensor([-0.5, -0.34, 0, 0.34, 0.5], dtype=float)

    def forward(self, output, target):
        # 在这里实现自定义的损失计算逻辑
        assert (
            output.size() == target.size()
        ), "the size of output should mathch the target"
        err_multi = torch.zeros_like(target, dtype=float)
        for i in range(target.size(0)):
            arg = target[i].argmax()
            delta = self.distance[arg]
            dis = abs(self.distance - delta)
            err_multi[i] = dis + 1
        loss = torch.pow((output - target) * err_multi, 2).sum() / target.size(0)
        return loss


def calc_error(y_pred, y_actual):
    with torch.no_grad():
        tot_loss = F.mse_loss(y_pred, y_actual)
        rmse = torch.sqrt(tot_loss).item()
        perc_loss = torch.mean(100.0 * torch.abs((y_pred - y_actual) / y_actual))
    return (tot_loss, rmse, perc_loss)


def train_vgg_lstm(
    model: torch.nn.Module,
    data: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.MSELoss,
    epochs=700,
):
    errors = []
    rmses = []
    accs = []
    for t in range(epochs):
        # Process each mini-batch in turn:
        for x, y_actual in data:
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(
                y_pred.to(dtype=torch.float), y_actual.to(dtype=torch.float)
            )
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Epoch ending, so now fit the coefficients based on all data:
        for x, y_actual in data:
            # Get the error rate for the whole batch:
            y_pred = model(x)
            mse, rmse, perc_loss = calc_error(y_pred, y_actual)
            pred = torch.zeros_like(y_pred)
            batch_num = y_pred.size(0)
            for i in range(batch_num):
                max_ind = y_pred[i].argmax()
                pred[i][max_ind] = 1
            corrects = (pred.data == y_actual.data).all(dim=1).sum()
            acc = corrects / batch_num * 100
            accs.append(acc)
            errors.append(perc_loss)
            rmses.append(rmse)
        # Print some progress information as the net is trained:
        if epochs < 30 or t % 10 == 0:
            print(
                "epoch {:4d}: RMSE={:.5f} ={:.2f}, Accracy={:.5f}%".format(
                    t,
                    sum(rmses) / len(rmses),
                    sum(errors) / len(errors),
                    sum(accs) / len(accs),
                )
            )
    torch.save(model.state_dict(), "vgg_lstm_model.pth")


if __name__ == "__main__":
    data = lstm_train_data("IC.CFX", batch_size, seq_len)
    model = VGG_LSTM(5, 20, seq_len, hidden_dim, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CustomLoss()
    # criterion = torch.nn.MSELoss(reduce="mean")
    train_vgg_lstm(model, data, optimizer, criterion, 700)
