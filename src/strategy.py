import numpy as np
import pandas as pd


def cross_sectional_reversal_signal(returns: pd.DataFrame) -> pd.DataFrame:
    demeaned = returns.sub(returns.mean(axis=1), axis=0)
    return -demeaned


def time_series_momentum_signal(returns: pd.DataFrame, lookback: int = 20,
                                skip: int = 1) -> pd.DataFrame:
    logret = np.log1p(returns)
    trailing = np.expm1(logret.shift(skip).rolling(lookback).sum())
    return trailing.sub(trailing.mean(axis=1), axis=0)


def _normalize_row(row: pd.Series) -> pd.Series:
    s = row - row.mean()                 # dollar-neutral
    denom = s.abs().sum()
    if denom == 0 or np.isnan(denom):
        return s * 0.0
    return s / denom                     # gross = 1


def to_weights(signal: pd.DataFrame) -> pd.DataFrame:
    return signal.apply(_normalize_row, axis=1)


def volatility_scale(signal: pd.DataFrame, returns: pd.DataFrame,
                     lookback: int = 20) -> pd.DataFrame:
    vol = returns.rolling(lookback).std()
    inv = 1.0 / vol.replace(0.0, np.nan)
    return signal * inv


def volume_filter(signal: pd.DataFrame, volume: pd.DataFrame,
                  lookback: int = 20, cap: float = 3.0) -> pd.DataFrame:
    mean = volume.rolling(lookback).mean()
    std = volume.rolling(lookback).std().replace(0.0, np.nan)
    z = (volume - mean) / std
    mult = 1.0 + z.clip(lower=0.0, upper=cap)   # amplify high-volume moves
    return signal * mult.fillna(1.0)


def no_trade_band(weights: pd.DataFrame, band: float = 0.02) -> pd.DataFrame:
    out = weights.copy()
    prev = None
    for i in range(len(weights)):
        target = weights.iloc[i]
        if prev is None:
            prev = target.fillna(0.0)
            out.iloc[i] = prev
            continue
        if target.isna().all():
            out.iloc[i] = prev          # data gap: hold the book, don't unwind
            continue
        target = target.fillna(0.0)
        held = prev.copy()
        update = (target - prev).abs() > band
        held[update] = target[update]
        out.iloc[i] = held
        prev = held
    return out


def combine_inverse_vol(ret_a: pd.Series, ret_b: pd.Series,
                        lookback: int = 30) -> pd.Series:
    vol_a = ret_a.rolling(lookback).std()
    vol_b = ret_b.rolling(lookback).std()
    inv_a = 1.0 / vol_a.replace(0.0, np.nan)
    inv_b = 1.0 / vol_b.replace(0.0, np.nan)
    total = inv_a + inv_b
    w_a = (inv_a / total).shift(1)   # lag weights: no lookahead
    w_b = (inv_b / total).shift(1)
    combined = w_a * ret_a + w_b * ret_b
    return combined
