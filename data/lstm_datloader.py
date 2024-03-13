import os
from typing import Literal
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE

from utils import read_env

env_path = Path(__file__).parent.parent / "env_vars.txt"
env = read_env(env_path)
os.environ.update(env)
input_dim = int(os.environ["INPUT_DIM"])


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


def data_to_zscore(data: pd.DataFrame) -> pd.DataFrame:
    features = data.drop(columns=["change1", "ts_code"])
    pcg = list(data["close"].pct_change())
    returns = features.iloc[:, 1:22].astype(float).apply(np.log).diff()
    indicaters = features.iloc[:, 22:].astype(float)
    dims = len(features.columns)
    for i in range(1, 22):
        features.iloc[:, i] = cal_zscore(returns.iloc[:, i - 1].values)
    for i in range(22, 53):
        features.iloc[:, i] = cal_zscore(indicaters.iloc[:, i - 26].values)
        pass
    pcg_df = pd.DataFrame({"pcg_zscore": cal_zscore(pcg)})
    data = pd.concat(
        [features.iloc[:-1, :], pcg_df.iloc[1:, :].reset_index(drop=True)],
        axis=1,
    )
    return data.iloc[1:, :]


def make_data(
    code: str, split_date: int = 20220913
) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_path = Path(__file__).parent / f"{code}.csv"
    fu_dat = pd.read_csv(file_path)
    data = data_to_zscore(fu_dat)
    train_data = (
        data[data["trade_date"] < split_date]
        .drop(columns=["trade_date"])
        .reset_index(drop=True)
    )
    test_data = (
        data[data["trade_date"] >= split_date]
        .drop(columns=["trade_date"])
        .reset_index(drop=True)
    )
    train_data.to_csv(Path(__file__).parent / f"{code}_train_data.csv", index=False)
    test_data.to_csv(Path(__file__).parent / f"{code}_test_data.csv", index=False)
    return train_data, test_data


def get_labled_data(
    data: pd.DataFrame,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    resample: bool = True,
) -> DataLoader:
    ros = SMOTE()
    x = torch.tensor(data.iloc[:, :-1].to_numpy(), dtype=torch.float32)
    y = mark_zscore(data.iloc[:, -1].values)
    y = torch.tensor(y, dtype=torch.float32)
    x = make_seqs(seq_len, x)
    x = x.view(x.size(0), -1).numpy()
    y = make_seqs(seq_len, y)[:, -1, :].numpy()
    if resample:
        x, y = ros.fit_resample(x, y)
    x = torch.tensor(x, dtype=torch.float32).view(-1, seq_len, input_dim)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def lstm_data(
    code: str,
    batch_size: int,
    seq_len: int,
    datype: Literal["train", "test"],
    shuffle: bool = True,
    split_data: int = 20220913,
) -> DataLoader:
    data_path = Path(__file__).parent / f"{code}_{datype}_data.csv"
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        if datype == "train":
            data, _ = make_data(code, split_data)
        else:
            _, data = make_data(code, split_data)
    data_loader = get_labled_data(data, seq_len, batch_size, shuffle)
    return data_loader


def make_seqs(seq_len: int, data: torch.Tensor):
    num_samp = data.size(0)
    return torch.stack([data[i : i + seq_len] for i in range(num_samp - seq_len + 1)])


def lstm_train_data(
    code: str, batch_size: int, seq_len: int, split_data: int = 20220913
):
    return lstm_data(code, batch_size, seq_len, "train", split_data=split_data)


def lstm_test_data(
    code: str,
    batch_size: int,
    seq_len: int,
    shuffle: bool = True,
    split_data: int = 20220913,
):
    return lstm_data(
        code, batch_size, seq_len, "test", shuffle=shuffle, split_data=split_data
    )


if __name__ == "__main__":
    # datald = lstm_train_data("IF.CFX", 64, 50)
    # dat = datald.dataset.tensors
    # print(dat[0].shape)
    make_data("IF.CFX")
