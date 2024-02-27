import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


from data.lstm_datloader import make_lstm_data
from model.convLstm import ConvLSTM
from model.cnn_lstm import Args, CNN_LSTM
from model.vgg_lstm import VGG_LSTM


batch_size = 64
input_dim = 20
hidden_dim = 100
seq_len = 50
num_layers = 1
class_num = 5
batch_first = True


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


def train_vgg_lstm(
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
            y_pred = model(x)
            y_actual = y_actual[:, -1, :]
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
            y_actual = y_actual[:, -1, :]
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


if __name__ == "__main__":
    data = make_lstm_data("IC.CFX", batch_size, seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss(reduction="sum")
    model = VGG_LSTM(5, 20, seq_len, hidden_dim)
    train_vgg_lstm(model, data, optimizer, criterion)
