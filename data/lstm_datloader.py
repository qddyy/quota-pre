import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def tag_zs(zs: float) -> list:
    if zs >= 0.069:
        return [0, 0, 0, 0, 1]
    elif 0.013 < zs < 0.069:
        return [0, 0, 0, 1, 0]
    elif -0.013 <= zs < 0.013:
        return [0, 0, 1, 0, 0]
    elif -0.069 <= zs < -0.013:
        return [0, 1, 0, 0, 0]
    else:
        return [1, 0, 0, 0, 0]


def cal_zscore(pcg: list):
    pcg = np.where(np.isinf(pcg), 0, pcg)
    zscores = [
        (pcg[i] - np.mean(pcg[1:])) / np.std(pcg[1:]) for i in range(1, len(pcg))
    ]
    zscores.insert(0, 0)
    return list(map(tag_zs, zscores))


def make_lstm_data(code: str, batch_size: int, seq_len: int) -> DataLoader:
    file_path = Path(__file__).parent / f"data/{code}.csv"
    fu_dat = pd.read_csv(file_path)
    x = fu_dat.drop(columns=["change1", "ts_code", "trade_date"])
    pcg = list(fu_dat["change1"].pct_change())
    pcg[0] = 0
    y = cal_zscore(pcg)
    x = torch.tensor(x.to_numpy(dtype=float), dtype=torch.float32)
    y = y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(make_seqs(seq_len, x), make_seqs(seq_len, y))
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader


def make_seqs(seq_len: int, data: torch.Tensor):
    num_samp = data.size(0)
    return torch.stack([data[i : i + seq_len] for i in range(num_samp - seq_len)])
