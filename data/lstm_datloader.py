import os
from typing import Literal
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE


def tag_zs(zs: float) -> list:
    if zs >= 1:
        return [0, 0, 0, 0, 1]
    elif 0.2 <= zs < 1:
        return [0, 0, 0, 1, 0]
    elif -0.2 < zs < 0.2:
        return [0, 0, 1, 0, 0]
    elif -1 < zs <= -0.2:
        return [0, 1, 0, 0, 0]
    else:
        return [1, 0, 0, 0, 0]


def cal_zscore(pcg: list):
    pcg = np.where(np.isinf(pcg), 0, pcg)
    zscores = [
        (pcg[i] - np.mean(pcg[1:])) / np.std(pcg[1:]) for i in range(1, len(pcg))
    ]
    zscores.insert(0, 0)
    return zscores


def mark_zscore(zscores: list):
    return list(map(tag_zs, zscores))


def make_data(code: str) -> pd.DataFrame:
    file_path = Path(__file__).parent / f"{code}.csv"
    fu_dat = pd.read_csv(file_path)
    features = fu_dat.drop(columns=["change1", "ts_code"])
    pcg = list(fu_dat["close"].pct_change())
    returns = features.iloc[:, 1:8].astype(float).apply(np.log).diff()
    indicaters = features.iloc[:, 8:-1].astype(float)
    # returns = features.iloc[:, 1:7].pct_change()
    for i in range(1, 8):
        features.iloc[:, i] = cal_zscore(returns.iloc[:, i - 1].values)
    for i in range(8, 20):
        features.iloc[:, i] = cal_zscore(indicaters.iloc[:, i - 8].values)
    pcg_df = pd.DataFrame({"pcg_zscore": cal_zscore(pcg)})
    data = pd.concat(
        [features.iloc[:-1, :], pcg_df.iloc[1:, :].reset_index(drop=True)],
        axis=1,
    )
    train_data = (
        data[data["trade_date"] < 20220913]
        .iloc[1:, :]
        .drop(columns=["trade_date"])
        .reset_index(drop=True)
    )
    test_data = (
        data[data["trade_date"] >= 20220913]
        .drop(columns=["trade_date"])
        .reset_index(drop=True)
    )
    train_data.to_csv(Path(__file__).parent / f"{code}_train_data.csv", index=False)
    test_data.to_csv(Path(__file__).parent / f"{code}_test_data.csv", index=False)
    return train_data, test_data


def lstm_data(
    code: str, batch_size: int, seq_len: int, datype: Literal["train", "test"]
) -> DataLoader:
    data_path = Path(__file__).parent / f"{code}_{datype}_data.csv"
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        if datype == "train":
            data, _ = make_data(code)
        else:
            _, data = make_data(code)
    ros = SMOTE()
    x = torch.tensor(data.iloc[:, :-1].to_numpy(), dtype=torch.float32)
    y = mark_zscore(data.iloc[:, -1].values)
    y = torch.tensor(y, dtype=torch.float32)
    x = make_seqs(seq_len, x)
    x = x.view(x.size(0), -1).numpy()
    y = make_seqs(seq_len, y)[:, -1, :].numpy()
    x_resampled, y_resampled = ros.fit_resample(x, y)
    x = torch.tensor(x_resampled, dtype=torch.float32).view(-1, seq_len, 20)
    y = torch.tensor(y_resampled, dtype=torch.float32)
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader


def make_seqs(seq_len: int, data: torch.Tensor):
    num_samp = data.size(0)
    return torch.stack([data[i : i + seq_len] for i in range(num_samp - seq_len)])


def lstm_train_data(code: str, batch_size: int, seq_len: int):
    return lstm_data(code, batch_size, seq_len, "train")


def lstm_test_data(code: str, batch_size: int, seq_len: int):
    return lstm_data(code, batch_size, seq_len, "test")


if __name__ == "__main__":
    datald = lstm_train_data("IF.CFX", 64, 50)
    for x, y in datald:
        print(x.shape, y.shape)
