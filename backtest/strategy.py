import os
import sys
import math
from pathlib import Path
from typing import Callable
from datetime import datetime, timedelta


import torch
from torch.nn.functional import normalize
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


def read_data(code: str) -> pd.DataFrame:
    data_path = Path(__file__).parent.parent / f"data/{code}_test_data.csv"
    if os.path.exists(data_path):
        test_data = pd.read_csv(data_path)
    else:
        _, test_data = make_data(code)
    return test_data


def read_orin_data(code: str) -> pd.DataFrame:
    file_path = Path(__file__).parent.parent / f"data/{code}.csv"
    fu_dat = pd.read_csv(file_path)
    features = fu_dat.drop(columns=["change1", "ts_code"])
    data = features.iloc[:-1, :]
    test_data = (
        data[data["trade_date"] >= 20220913]
        .drop(columns=["trade_date"])
        .reset_index(drop=True)
    )
    return test_data


def make_vgg_data(code: str, seq_len: int) -> torch.Tensor:
    test_data = read_data(code)
    features = torch.tensor(test_data.iloc[:, :-1].to_numpy(), dtype=torch.float32)
    features = make_seqs(seq_len, features)
    return features


def make_gbdt_data(code: str, seq_len: int):
    _, _, test_data, _ = split_data(code, seq_len, False)
    return test_data.ffill()


def unilize(signals: torch.Tensor) -> torch.Tensor:
    normal = (signals - signals.min()) / (signals.max() - signals.min())
    return normal / normal.sum()


def execut_signal(
    code: str,
    account: futureAccount,
    weight: torch.Tensor,
    signals: torch.Tensor,
    price: float,
):
    # volumes_rate = (unilize(signals) * weight).sum().item()
    arg = signals.argmax().item()
    volumes_rate = weight[arg].item()
    account.order_to(code, volumes_rate, price)


def generate_signal(data, model):
    signals = model(data)
    return signals


class strategy:
    model: Callable
    weight: torch.Tensor
    has_signal: bool
    pre_times: int
    code: str
    seq_len: int
    orin_data: pd.DataFrame
    test_data: pd.DataFrame
    account: futureAccount
    portfolio_values: list

    def __init__(self, code: str, seq_len: int, test_data: pd.DataFrame) -> None:
        self.code = code
        self.pre_times = 0
        self.signals = None
        self.has_signal = False
        self.seq_len = seq_len
        self.test_data = test_data
        self.orin_data = read_orin_data(code)
        self.win_times = []
        self.odds = {"win": [], "loss": []}
        self.portfolio_values = []
        self.weight = torch.tensor([-0.5, -0.2, 0.0, 0.2, 0.5], dtype=torch.float32)
        self.account = futureAccount(current_date="2022-09-13", base=10000000, pool={})

    def excute_stratgy(self, signal_gerater: Callable, model: Callable | None = None):
        for i in range(len(self.orin_data)):
            self.account.update_date(1)
            price = self.orin_data.loc[i, ["close"]].item()
            self.daily_settle(price)
            if self.has_signal:
                execut_signal(self.code, self.account, self.weight, self.signals, price)
                self.has_signal = False
            self.account.update_price({self.code: price})
            self.portfolio_values.append(self.account.portfolio_value)
            if (i + 1) >= self.seq_len and i <= len(self.orin_data) - self.seq_len:
                self.signals = signal_gerater(self.test_data[self.pre_times], model)
                self.pre_times += 1
                self.has_signal = True

    def daily_settle(self, current_price: float):
        today = self.account.current_date
        yesterday = roll_date(today)
        if self.account.transactions and self.account.pool:
            if yesterday in self.account.transactions.keys():
                transacton = self.account.transactions[yesterday][-1]
                old_price = transacton["price"]
                volume = transacton["volume"]
                self.account.order(self.code, -volume, current_price)
                new_volum = self.account.transactions[today][-1]["volume"]
                returns = -new_volum * current_price
                pay = volume * old_price
                intrest = returns - pay
                if intrest > 0:
                    self.win_times.append(1)
                    self.odds["win"].append(intrest)
                else:
                    self.win_times.append(0)
                    self.odds["loss"].append(abs(intrest))


def lstm_sig_gener(data, model) -> torch.Tensor:
    return generate_signal(data.unsqueeze(0), model)


def roll_date(date: str):
    date_format = "%Y-%m-%d"
    old_date = datetime.strptime(date, date_format)
    new_date = (old_date + timedelta(days=-1)).strftime(date_format)
    return new_date


def vgg_lstm_strategy(code: str, seq_len: int):
    model_path = Path(__file__).parent.parent / "vgg_lstm_model.pth"
    model = VGG_LSTM(5, 20, 50, 100)
    model.load_state_dict(torch.load(model_path))
    test_data = make_vgg_data(code, seq_len)
    executer = strategy(code, seq_len, test_data)
    executer.excute_stratgy(lstm_sig_gener, model)
    portfolio_values = executer.portfolio_values
    win_rate = sum(executer.win_times) / len(executer.win_times)
    odds = sum(list(executer.odds["win"])) / sum(list(executer.odds["loss"]))
    # print(executer.account.transactions)
    return [v / portfolio_values[0] for v in portfolio_values], win_rate, odds


def gbdt_sig_gener(data, model) -> torch.Tensor:
    s = generate_signal([data.to_numpy()], model).squeeze()
    signals = torch.tensor(s, dtype=torch.float32)
    return signals


def random_gener(data, model) -> torch.Tensor:
    return torch.randn(5)


def gbdt_strategy(code: str, seq_len: int):
    model_path = Path(__file__).parent.parent / "model.txt"
    model = lgb.Booster(model_file=model_path).predict
    test_data = make_gbdt_data(code, seq_len).iloc
    executer = strategy(code, seq_len, test_data)
    executer.excute_stratgy(gbdt_sig_gener, model)
    portfolio_values = executer.portfolio_values
    win_rate = sum(executer.win_times) / len(executer.win_times)
    odds = sum(list(executer.odds["win"])) / sum(list(executer.odds["loss"]))
    return [v / portfolio_values[0] for v in portfolio_values], win_rate, odds


def random_strategy(code: str, seq_len: int):
    test_data = make_gbdt_data(code, seq_len)
    executer = strategy(code, seq_len, test_data)
    executer.excute_stratgy(random_gener)
    portfolio_values = executer.portfolio_values
    win_rate = sum(executer.win_times) / len(executer.win_times)
    odds = sum(list(executer.odds["win"])) / sum(list(executer.odds["loss"]))
    return [v / portfolio_values[0] for v in portfolio_values], win_rate, odds


def bench_mark(code: str) -> pd.Series:
    data = read_orin_data(code)
    return data["close"] / data["close"][0]


if __name__ == "__main__":
    gbdt_result, gbdt_wrate, gbdt_odds = gbdt_strategy("IC.CFX", 50)
    vgg_lstm_result, lstm_wrate, lstm_odds = vgg_lstm_strategy("IC.CFX", 50)
    random_result, rand_wrate, rand_odds = random_strategy("IC.CFX", 50)
    bench_result = list(bench_mark("IC.CFX").values)
    returns = pd.DataFrame(
        {
            "lstm": vgg_lstm_result,
            "gbdt": gbdt_result,
            "random": random_result,
            "bench": bench_result,
        }
    )
    rates = pd.DataFrame(
        {
            "strategy": ["lstm", "gbdt", "random"],
            "win_rates": [lstm_wrate, gbdt_wrate, rand_wrate],
            "odds": [lstm_odds, gbdt_odds, rand_odds],
        }
    )
    print(rates)
    returns.dropna().plot()
    plt.show()
