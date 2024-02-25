import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from data.lstm_datloader import cal_zscore, make_seqs, make_lstm_data
from model.convLstm import ConvLSTM
from model.cnn_lstm import Args, CNN_LSTM


batch_size = 128
input_dim = 20
hidden_dim = 60
seq_len = 30
height = 1
width = 1
kernel_size = (3, 3)
num_layers = 1
class_num = 5
batch_first = True  # 如果您的输入数据的第一个维度是 batch 维度，则设置为 True
bias = True
return_all_layers = False  # 如果只需要最后一层的输出，则设置为 False

args = Args(
    input_dim,
    hidden_dim,
    num_layers,
    batch_first,
    batch_size,
    class_num,
    batch_size // 2,
    kernel_size,
    0.5,
)

model = ConvLSTM(
    input_dim, hidden_dim, kernel_size, num_layers, batch_first, bias, return_all_layers
)

cnn_lstm_model = CNN_LSTM(args)


def add_data(data: torch.Tensor, batch_size: int):
    if data.size(0) < batch_size:
        row_num = batch_size - data.size(0)
        zero = torch.zeros((row_num, data.size(1), data.size(2)))
        data = torch.cat([data, zero], dim=0)
        return data
    else:
        return data


def calc_error(y_pred, y_actual):
    with torch.no_grad():
        tot_loss = F.mse_loss(y_pred, y_actual)
        rmse = torch.sqrt(tot_loss).item()
        perc_loss = torch.mean(100.0 * torch.abs((y_pred - y_actual) / y_actual))
    return (tot_loss, rmse, perc_loss)


def train_cnn_lstm(
    model: torch.nn.Module,
    data: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.MSELoss,
    epochs=500,
):
    errors = []
    rmses = []
    accs = []
    for t in range(epochs):
        # Process each mini-batch in turn:
        for x, y_actual in data:
            if x.size(0) < batch_size:
                x, y_actual = add_data(x, batch_size), add_data(y_actual, batch_size)
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
            if x.size(0) < batch_size:
                x, y_actual = add_data(x, batch_size), add_data(y_actual, batch_size)
            # Get the error rate for the whole batch:
            y_pred = model(x)
            mse, rmse, perc_loss = calc_error(y_pred, y_actual)
            pred = torch.zeros_like(y_pred)
            for i in range(batch_size):
                _, max_ind = torch.max(y_pred[i], dim=1)
                pred[i][:, max_ind] = 1
            corrects = (pred.data == y_actual.data).all(dim=2).sum()
            acc = corrects / (batch_size * seq_len) * 100
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


if __name__ == "__main__":
    data = make_lstm_data("IC.CFX", batch_size, seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss(reduction="sum")
    train_cnn_lstm(cnn_lstm_model, data, optimizer, criterion)
