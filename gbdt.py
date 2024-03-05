from typing import Literal
from pathlib import Path
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from data.lstm_datloader import make_data, tag_zs

num_round = 10
windows = 50


def trans_class_num(cls: list):
    return cls.index(max(cls))


def flat_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, windows: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_feats = []
    train_tages = []
    test_feats = []
    test_tages = []
    for i in range(len(train_data) - windows + 1):
        train_feats.append(train_data.iloc[i : i + windows, :-1].values.flatten())
        train_tages.append(train_data.iloc[i + windows - 1, -1])
    for i in range(len(test_data) - windows + 1):
        test_feats.append(test_data.iloc[i : i + windows, :-1].values.flatten())
        test_tages.append(test_data.iloc[i + windows - 1, -1])
    return (
        pd.DataFrame(train_feats),
        pd.DataFrame(train_tages),
        pd.DataFrame(test_feats),
        pd.DataFrame(test_tages),
    )


def split_data(
    code: Literal["IC.CFX", "IF.CFX", "IH.CFX", "IM.CFX"],
    windows: int,
    resample: bool = True,
    split_date: int = 20220913,
):
    # 重采样
    ros = RandomOverSampler(random_state=42)
    train_data, test_data = make_data(code, split_date)
    x_train, y_train, x_test, y_test = flat_data(train_data, test_data, windows)
    y_train = (
        y_train.iloc[:, -1].apply(tag_zs).apply(trans_class_num).reset_index(drop=True)
    )
    y_test = (
        y_test.iloc[:, -1].apply(tag_zs).apply(trans_class_num).reset_index(drop=True)
    )
    if resample:
        x_train, y_train = ros.fit_resample(x_train, y_train)
        x_test, y_test = ros.fit_resample(x_test, y_test)
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


def test_gbdt(code: str, seq_len: int, split_date: int = 20220913):
    *_, x_test, y_test = split_data(code, seq_len, split_date)
    # 模型效果评估
    model_path = Path(__file__).parent / f"{code}_gbdt_model.txt"
    bst = lgb.Booster(model_path)
    y_pred = bst.predict(x_test)
    y_pred = pd.Series(map(lambda x: x.argmax(), y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    y_test.index = range(0, len(y_test.index))
    y_pred = y_pred
    print(sum(y_test == y_pred) / len(y_pred))
    print(accuracy)


if __name__ == "__main__":
    test_gbdt("IF.CFX", windows)
