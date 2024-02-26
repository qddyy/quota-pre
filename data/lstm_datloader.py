import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import RandomOverSampler


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
    features = fu_dat.drop(columns=["change1", "ts_code", "trade_date"])
    pcg = list(fu_dat["close"].pct_change())
    # returns = features.iloc[:, :7].astype(float).apply(np.log).diff()
    # returns = features.iloc[:, :7].pct_change()
    # for i in range(7):
    #     features.iloc[:, i] = cal_zscore(returns.iloc[:, i].values)
    pcg_df = pd.DataFrame({"pcg_zscore": cal_zscore(pcg)})
    data = pd.concat(
        [features.iloc[:-1, :], pcg_df.iloc[1:, :].reset_index(drop=True)],
        axis=1,
        ignore_index=True,
    )
    return data.iloc[1:, :].reset_index(drop=True)


def make_lstm_data(code: str, batch_size: int, seq_len: int) -> DataLoader:
    data = make_data(code)
    ros = RandomOverSampler(random_state=42)
    x = data.iloc[:, :-1].to_numpy(dtype=float)
    y = np.array(mark_zscore(data.iloc[:, -1]))
    x_resampled, y_resampled = ros.fit_resample(x, y)
    x = torch.tensor(x_resampled, dtype=torch.float32)
    y = torch.tensor(y_resampled, dtype=torch.float32)
    dataset = TensorDataset(make_seqs(seq_len, x), make_seqs(seq_len, y))
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader


def make_seqs(seq_len: int, data: torch.Tensor):
    num_samp = data.size(0)
    return torch.stack([data[i : i + seq_len] for i in range(num_samp - seq_len)])


if __name__ == "__main__":
    print(make_lstm_data("IC.CFX", 32, 8))
