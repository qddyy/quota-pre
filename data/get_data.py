from datetime import date

import tushare as ts
import pandas as pd
import talib as ta


pro = ts.pro_api()

today = int(date.today().strftime("%Y%m%d"))
fut_codes = ["IH.CFX", "IF.CFX", "IC.CFX", "IM.CFX"]
index_codes = ["000016.SH", "000300.SH", "000905.SH", "000852.SH"]
fut_info = {
    "IH.CFX": ["000016.SH", "510050.SH", 20160224, today],
    "IF.CFX": ["000300.SH", "159919.SZ", 20160224, today],
    "IC.CFX": ["000905.SH", "159922.SZ", 20160224, today],
    "IM.CFX": ["000852.SH", "159845.SZ", 20221130, today],
}
indics = [
    "ma",
    "ema",
    "dema",
    "HT_TRENDLINE",
    "kama",
    "tema",
    "wma",
    "macd",
    "CCI",
    "DX",
    "MINUS_DI",
    "PLUS_DI",
    "MIDPOINT",
    "TRIMA",
]


def get_fu_data(
    code: str, add_ind: bool = False, add_etf: bool = False
) -> pd.DataFrame:
    start_date = fut_info[code][-2]
    end_date = fut_info[code][-1]
    fu_dat = pro.fut_daily(ts_code=code, start_date=start_date, end_date=end_date)
    fu_dat = fu_dat[
        [
            "ts_code",
            "trade_date",
            "pre_close",
            "pre_settle",
            "open",
            "high",
            "low",
            "close",
            "settle",
            "vol",
            "oi",
            "change1",
        ]
    ]
    if add_ind:
        fu_dat = merge_ind(fu_dat, code)
    if add_etf:
        fu_dat = merge_etf(fu_dat, code)
    return fu_dat


def get_fu_single_indi(
    code: str, indis: list[str], add_ind: bool = False, add_etf: bool = False
) -> pd.DataFrame:
    fu_dat = get_fu_data(code, add_ind, add_etf)
    for ind in indis:
        ind = ind.upper()
        ind_op = getattr(ta, ind)
        if ind in ["MACD", "MACDFIX", "MACDEXT"]:
            ind_dat, *_ = ind_op(fu_dat["close"].values[::-1])
            if add_ind:
                ind_index, *_ = ind_op(fu_dat["ind_close"].values[::-1])
            if add_etf:
                ind_etf, *_ = ind_op(fu_dat["etf_close"].values[::-1])
        elif ind in ["ADX", "ADXR", "CCI", "DX", "MINUS_DI", "PLUS_DI"]:
            ind_dat = ind_op(
                high=fu_dat["high"].values[::-1],
                low=fu_dat["low"].values[::-1],
                close=fu_dat["close"].values[::-1],
            )
            if add_ind:
                ind_index = ind_op(
                    high=fu_dat["ind_high"].values[::-1],
                    low=fu_dat["ind_low"].values[::-1],
                    close=fu_dat["ind_close"].values[::-1],
                )
            if add_etf:
                ind_etf = ind_op(
                    high=fu_dat["etf_high"].values[::-1],
                    low=fu_dat["etf_low"].values[::-1],
                    close=fu_dat["etf_close"].values[::-1],
                )
        else:
            ind_dat = ind_op(fu_dat["close"].values[::-1])
            if add_ind:
                ind_index = ind_op(fu_dat["ind_close"].values[::-1])
            if add_etf:
                ind_etf = ind_op(fu_dat["etf_close"].values[::-1])
        ind_df = pd.DataFrame({ind: ind_dat[::-1]})
        if add_ind:
            index_df = pd.DataFrame({f"{ind}_index": ind_index[::-1]})
            ind_df = pd.concat([ind_df, index_df], axis=1)
        if add_etf:
            etf_df = pd.DataFrame({f"{ind}_etf": ind_etf[::-1]})
            ind_df = pd.concat([ind_df, etf_df], axis=1)
        fu_dat = pd.concat([fu_dat, ind_df], axis=1)
    return fu_dat


def save_fut_data(
    codes: list[str],
    indis: list[str] | None = None,
    add_ind: bool = False,
    add_etf: bool = False,
) -> None:
    if indis is None:
        for co in codes:
            data_name = f"./data/{co}.csv"
            fu_dat = get_fu_data(co, add_ind, add_etf)
            fu_dat.to_csv(data_name, index=False)
    else:
        for co in codes:
            data_name = f"./data/{co}.csv"
            fu_dat = get_fu_single_indi(co, indis, add_ind, add_etf).dropna()
            fu_dat[::-1].to_csv(data_name, index=False)


def get_fu_index(fu_code: str) -> pd.DataFrame:
    code = fut_info[fu_code][0]
    start_date = fut_info[fu_code][-2]
    end_date = fut_info[fu_code][-1]
    ind_dat = pro.index_daily(
        ts_code=code, start_date=start_date, end_date=end_date
    ).drop(columns=["amount", "ts_code"])
    ind_dat.columns = ["ind_" + v if v != "trade_date" else v for v in ind_dat.columns]
    return ind_dat


def merge_ind(data: pd.DataFrame, fu_code: str) -> pd.DataFrame:
    ind = get_fu_index(fu_code)
    merge_dat = pd.merge(data, ind, on="trade_date")
    return merge_dat


def merge_etf(data: pd.DataFrame, fu_code: str) -> pd.DataFrame:
    ind = get_fu_etf(fu_code)
    merge_dat = pd.merge(data, ind, on="trade_date")
    return merge_dat


def get_fu_etf(fu_code: str) -> pd.DataFrame:
    code = fut_info[fu_code][1]
    start_date = fut_info[fu_code][-2]
    end_date = fut_info[fu_code][-1]
    etf_dat = pro.fund_daily(
        ts_code=code, start_date=start_date, end_date=end_date
    ).drop(columns=["amount", "ts_code"])
    etf_dat.columns = ["etf_" + v if v != "trade_date" else v for v in etf_dat.columns]
    return etf_dat


if __name__ == "__main__":
    save_fut_data(fut_codes, indics, True, True)
