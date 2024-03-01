import os
import sys
import math
from pathlib import Path


import torch
import pandas as pd
import pandas as np
import matplotlib.pyplot as plt
import lightgbm as lgb

from backtest.schema import futureAccount
from data.lstm_datloader import make_data, make_seqs
from model.vgg_lstm import VGG_LSTM
from gbdt import split_data


class tradeSignal:
    HARD_BUY: float = 0.5
    LITTLE_BUY: float = 0.2
    FLAT: float = 0.0
    LITTLE_SELL: float = -0.2
    HARD_SELL: float = -0.5


# def make_seqs(seq_len: int, data: torch.Tensor):
#     num_samp = data.size(0)
#     stack = []
#     for i in range(0, num_samp, seq_len):
#         stack.append(data[i : min(i + seq_len, num_samp)])
#     add = torch.zeros_like(stack[0])
#     last_seq = stack[-1].size(0)
#     if last_seq < seq_len:
#         add[:last_seq, :] = stack[-1]
#         for i in range(last_seq, seq_len):
#             add[i, :] = stack[-1][-1, :]
#     stack[-1] = add
#     return torch.stack(stack, dim=0)


def read_data(code: str) -> pd.DataFrame:
    data_path = Path(__file__).parent.parent / f"data/{code}_test_data.csv"
    if os.path.exists(data_path):
        test_data = pd.read_csv(data_path)
    else:
        _, test_data = make_data(code)
    return test_data


def make_vgg_data(code: str, seq_len: int) -> torch.Tensor:
    test_data = read_data(code)
    features = torch.tensor(test_data.iloc[:, :-1].to_numpy(), dtype=torch.float32)
    features = make_seqs(seq_len, features)
    return features


def flat_data(data: pd.DataFrame, windows: int) -> pd.DataFrame:
    test_data = []
    num_samp = len(data)
    for i in range(0, num_samp, windows):
        test_data.append(data[i : min(i + windows, num_samp)].values.flatten())
    return pd.DataFrame(test_data)


def make_gbdt_data(code: str, seq_len: int):
    _, _, test_data, _ = split_data(code, seq_len, False)
    return test_data.fillna(method="ffill")


def execut_signal(
    code: str,
    account: futureAccount,
    weight: torch.Tensor,
    signals: torch.Tensor,
    price: float,
):
    volumes_rate = (signals * weight).sum().item()
    print(volumes_rate)
    account.order_to(code, volumes_rate, price)


def generate_signal(data, model):
    signals = model(data)
    return signals


def vgg_lstm_strategy(code: str, seq_len: int):
    pre_times = 0
    signals = None
    portfolio_values = []
    model_path = Path(__file__).parent.parent / "vgg_lstm_model.pth"
    model = VGG_LSTM(5, 20, 50, 100)
    model.load_state_dict(torch.load(model_path))
    has_siganl = False
    weight = torch.tensor([-0.5, -0.2, 0.0, 0.2, 0.5], dtype=torch.float32)
    account = futureAccount(current_date="2022-09-13")
    data = read_data(code)
    test_data = make_vgg_data(code, seq_len)
    for i in range(len(data)):
        account.update_date(1)
        price = data.loc[i, ["close"]].item()
        if has_siganl:
            execut_signal(code, account, weight, signals, price)
            has_siganl = False
        account.update_price({code: price})
        portfolio_values.append(account.portfolio_value)
        if (i + 1) >= seq_len and i <= len(data) - seq_len:
            signals = generate_signal(test_data[pre_times].unsqueeze(0), model)
            pre_times += 1
            has_siganl = True
    return portfolio_values


def gbdt_strategy(code: str, seq_len: int):
    pre_times = 0
    signals = None
    portfolio_values = []
    model_path = Path(__file__).parent.parent / "model.txt"
    model = lgb.Booster(model_file=model_path).predict
    has_siganl = False
    weight = torch.tensor([-0.5, -0.2, 0.0, 0.2, 0.5], dtype=torch.float32)
    account = futureAccount(current_date="2022-09-13")
    data = read_data(code)
    test_data = make_gbdt_data(code, seq_len)
    print(test_data.shape)
    for i in range(len(data)):
        account.update_date(1)
        price = data.loc[i, ["close"]].item()
        if has_siganl:
            execut_signal(code, account, weight, signals, price)
            has_siganl = False
        account.update_price({code: price})
        portfolio_values.append(account.portfolio_value)
        if (i + 1) % seq_len == 0 or (i + 1) == len(data):
            signals = torch.tensor(
                generate_signal(test_data[pre_times], model), dtype=torch.float32
            )
            pre_times += 1
            has_siganl = True
    return portfolio_values


if __name__ == "__main__":
    result = gbdt_strategy("IC.CFX", 50)
    df = pd.DataFrame(result).iloc[:, 0] / pd.DataFrame(result).iloc[0, 0].item()
    df.dropna().plot()
    plt.show()
