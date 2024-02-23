import tushare as ts
import pandas as pd
import talib as ta


pro = ts.pro_api()

fut_codes = ["IH.CFX", "IF.CFX", "IC.CFX", "IM.CFX"]
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
]


def get_fu_data(code: str) -> pd.DataFrame:
    etf_fu = pro.fut_daily(ts_code=code, start_date="20100101", end_date="20231231")
    etf_fu = etf_fu[
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
            "change1",
        ]
    ]
    return etf_fu


def get_fu_single_indi(code: str, indis: list[str]) -> pd.DataFrame:
    etf_fu = get_fu_data(code)
    for ind in indis:
        ind = ind.upper()
        ind_op = getattr(ta, ind)
        if ind in ["MACD", "MACDFIX", "MACDEXT"]:
            ind_dat, *_ = ind_op(etf_fu["close"].values[::-1])
        elif ind in ["ADX", "ADXR", "CCI", "DX", "MINUS_DI", "PLUS_DI"]:
            ind_dat = ind_op(
                high=etf_fu["high"].values[::-1],
                low=etf_fu["low"].values[::-1],
                close=etf_fu["close"].values[::-1],
            )
        else:
            ind_dat = ind_op(etf_fu["close"].values[::-1])
        ind_df = pd.DataFrame({ind: ind_dat[::-1]})
        etf_fu = pd.concat([etf_fu, ind_df], axis=1)
    return etf_fu


def save_fut_data(codes: list[str], indis: list[str] | None = None) -> None:
    if indis is None:
        for co in codes:
            data_name = f"./data/{co}.csv"
            fu_dat = get_fu_data(co)
            fu_dat.to_csv(data_name, index=False)
    else:
        for co in codes:
            data_name = f"./data/{co}.csv"
            fu_dat = get_fu_single_indi(co, indis).dropna()
            fu_dat[::-1].to_csv(data_name, index=False)


if __name__ == "__main__":
    save_fut_data(codes=fut_codes, indis=indics)
