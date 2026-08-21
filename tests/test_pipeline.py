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
    volume = pd.DataFrame(rng.lognormal(10, 1, (n_days, n_coins)),
                          index=idx, columns=cols)
    return returns, volume


def test_reversal_returns_has_expected_shape_and_no_lookahead():
    returns, volume = _synthetic()
    r = pipeline.reversal_returns(returns, volume)
    assert isinstance(r, pd.Series)
    assert len(r) == len(returns)
    # first realized day must be NaN/0 because weights are shifted by 1
    assert pd.isna(r.iloc[0]) or r.iloc[0] == 0.0


def test_reversal_is_profitable_on_reversal_data():
    returns, volume = _synthetic()
    r = pipeline.reversal_returns(returns, volume, cost_bps=0.0).dropna()
    assert r.mean() > 0  # the constructed reversal edge is captured


def test_split_index_is_chronological_70_30():
    idx = pd.RangeIndex(100)
    train, test = pipeline.split_index(idx, train_frac=0.7)
    assert len(train) == 70 and len(test) == 30
    assert train.max() < test.min()


def test_select_momentum_lookback_returns_member_of_grid():
    returns, volume = _synthetic()
    grid = [5, 10, 20]
    best = pipeline.select_momentum_lookback(returns, volume, grid)
    assert best in grid


def test_build_combined_returns_three_aligned_series():
    returns, volume = _synthetic()
    rev, mom, comb = pipeline.build_combined(returns, volume, mom_lookback=10)
    assert len(rev) == len(mom) == len(comb) == len(returns)
