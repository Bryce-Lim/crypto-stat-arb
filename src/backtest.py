import pandas as pd


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    dw = weights.diff()
    if len(weights) > 0:
        dw.iloc[0] = weights.iloc[0]  # initial build from a flat book
    return dw.abs().sum(axis=1)


def run_backtest(weights: pd.DataFrame, returns: pd.DataFrame,
                 cost_bps: float = 20.0) -> pd.Series:
    gross = (weights * returns).sum(axis=1)
    turnover = compute_turnover(weights)
    net = gross - turnover * (cost_bps / 1e4)
    return net
