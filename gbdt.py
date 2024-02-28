from typing import Literal
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from data.lstm_datloader import make_data, tag_zs

num_round = 10


def trans_class_num(cls: list):
    return cls.index(max(cls))


def split_data(code: Literal["IC.CFX", "IF.CFX", "IH.CFX", "IM.CFX"]):
    # 重采样
    ros = RandomOverSampler(random_state=42)
    train_data, test_data = make_data("IC.CFX")
    x_train = train_data.iloc[:, :-1].reset_index(drop=True)
    y_train = (
        train_data.iloc[:, -1]
        .apply(tag_zs)
        .apply(trans_class_num)
        .reset_index(drop=True)
    )
    x_test = test_data.iloc[:, :-1].reset_index(drop=True)
    y_test = (
        test_data.iloc[:, -1]
        .apply(tag_zs)
        .apply(trans_class_num)
        .reset_index(drop=True)
    )
    x_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    x_test_resampled, y_test_resampled = ros.fit_resample(x_test, y_test)
    return x_resampled, y_resampled, x_test_resampled, y_test_resampled


x_train, y_train, x_test, y_test = split_data("IC.CFX")

# 模型训练
train_data = lgb.Dataset(x_train, label=y_train)
params = {
    "num_leaves": 31,
    "num_trees": 100,
    "objective": "multiclass",
    "num_class": 5,
}
bst = lgb.train(params, train_data, num_round)
bst.save_model("model.txt")

# 模型效果评估
y_pred = bst.predict(x_test)
y_pred = pd.Series(map(lambda x: x.argmax(), y_pred))
accuracy = accuracy_score(y_test, y_pred)
y_test.index = range(0, len(y_test.index))
y_pred = y_pred
print(sum(y_test == y_pred) / len(y_pred))
print(accuracy)
