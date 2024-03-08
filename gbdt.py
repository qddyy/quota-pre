import os
from typing import Literal
from pathlib import Path
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from data.lstm_datloader import make_data, tag_zs
from utils import read_env

num_round = 10
env_path = Path(__file__).parent / "env_vars.txt"
os.environ.update(read_env(env_path))
windows = int(os.environ["SEQ_LEN"])


def trans_class_num(cls: list):
    return cls.index(max(cls))


def flatten(data: pd.DataFrame, windows: int):
    feats = []
    targets = []
    for i in range(len(data) - windows + 1):
        feats.append(data.iloc[i : i + windows, :-1].values.flatten())
        targets.append(data.iloc[i + windows - 1, -1])
    return pd.DataFrame(feats), pd.DataFrame(targets)


def flat_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, windows: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_feats, train_tages = flatten(train_data, windows)
    test_feats, test_tages = flatten(test_data, windows)
    return train_feats, train_tages, test_feats, test_tages


def gbdt_labled_data(data: pd.DataFrame, windows: int, resample: bool = True):
    ros = RandomOverSampler(random_state=42)
    feats, targets = flatten(data, windows)
    targets = (
        targets.iloc[:, -1].apply(tag_zs).apply(trans_class_num).reset_index(drop=True)
    )
    if resample:
        feats, targets = ros.fit_resample(feats, targets)
    return feats, targets


def split_data(
    code: Literal["IC.CFX", "IF.CFX", "IH.CFX", "IM.CFX"],
    windows: int,
    resample: bool = True,
    split_date: int = 20220913,
):
    # 重采样
    train_data, test_data = make_data(code, split_date)
    x_train, y_train = gbdt_labled_data(train_data, windows, resample)
    x_test, y_test = gbdt_labled_data(test_data, windows, resample)
    return x_train, y_train, x_test, y_test


def train_gbdt(code: str, seq_len: int, split_date: int = 20220913):
    x_train, y_train, *_ = split_data(code, seq_len, split_date)

    # 模型训练
    train_data = lgb.Dataset(x_train, label=y_train)
    params = {
        "num_leaves": 31,
        "num_trees": 100,
        "metric": "multi_error",
        "objective": "multiclass",
        "num_class": 5,
    }
    bst = lgb.train(params, train_data, num_round)
    bst.save_model(f"{code}_gbdt_model.txt")
    return bst


def test_gbdt(code: str, seq_len: int, split_date: int = 20220913):
    *_, x_test, y_test = split_data(code, seq_len, split_date)
    # 模型效果评估
    model_path = Path(__file__).parent / f"{code}_gbdt_model.txt"
    bst = lgb.Booster(model_file=model_path)
    y_pred = bst.predict(x_test)
    y_pred = pd.Series(map(lambda x: x.argmax(), y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    y_test.index = range(0, len(y_test.index))
    y_pred = y_pred
    print(sum(y_test == y_pred) / len(y_pred))
    print(accuracy)


if __name__ == "__main__":
    train_gbdt("IC.CFX", windows)
    test_gbdt("IC.CFX", windows)
