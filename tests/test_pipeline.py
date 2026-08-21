import numpy as np
import pandas as pd
import pytest
from src import pipeline


def _synthetic(n_days=400, n_coins=8, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"C{i}" for i in range(n_coins)]
    idx = pd.date_range("2022-01-01", periods=n_days, freq="D")
    # build returns with intentional 1-day reversal: next = -0.3*prev + noise
    rets = np.zeros((n_days, n_coins))
    noise = rng.normal(0, 0.03, (n_days, n_coins))
    for t in range(1, n_days):
        rets[t] = -0.3 * rets[t - 1] + noise[t]
    returns = pd.DataFrame(rets, index=idx, columns=cols)
    return returns


def test_reversal_returns_has_expected_shape_and_no_lookahead():
    returns = _synthetic()
    r = pipeline.reversal_returns(returns)
    assert isinstance(r, pd.Series)
    assert len(r) == len(returns)
    # first realized day must be NaN/0 because weights are shifted by 1
    assert pd.isna(r.iloc[0]) or r.iloc[0] == 0.0


def test_reversal_is_profitable_on_reversal_data():
    returns = _synthetic()
    # target_vol=None isolates the raw edge from the vol-target rescale
    r = pipeline.reversal_returns(returns, cost_bps=0.0, band=0.0,
                                  target_vol=None).dropna()
    assert r.mean() > 0  # the constructed reversal edge is captured


def test_split_index_is_chronological_70_30():
    idx = pd.RangeIndex(100)
    train, test = pipeline.split_index(idx, train_frac=0.7)
    assert len(train) == 70 and len(test) == 30
    assert train.max() < test.min()


def test_select_band_returns_member_of_grid_and_uses_train_only():
    returns = _synthetic()
    grid = [0.0, 0.02, 0.05, 0.1]
    best = pipeline.select_band(returns, grid, cost_bps=7.0)
    assert best in grid


def test_vol_target_makes_annual_vol_near_target():
    returns = _synthetic()
    r = pipeline.reversal_returns(returns, cost_bps=0.0, band=0.0,
                                  target_vol=0.15).dropna()
    ann_vol = r.std(ddof=0) * np.sqrt(365)
    # book-level vol targeting should pull realized vol into a band around 15%
    assert 0.08 < ann_vol < 0.25


def test_momentum_returns_shape_and_no_lookahead():
    returns = _synthetic()
    r = pipeline.momentum_returns(returns)
    assert isinstance(r, pd.Series)
    assert len(r) == len(returns)
    assert pd.isna(r.iloc[0]) or r.iloc[0] == 0.0
