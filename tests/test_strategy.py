import numpy as np
import pandas as pd
import pytest
from src import strategy


def _rets(vals):
    return pd.DataFrame(vals, columns=["A", "B", "C"])


def test_reversal_is_negative_of_demeaned_return():
    r = _rets([[0.03, 0.00, -0.03]])
    sig = strategy.cross_sectional_reversal_signal(r)
    # mean 0 -> signal = -returns
    assert sig.iloc[0].tolist() == pytest.approx([-0.03, 0.0, 0.03])


def test_to_weights_dollar_neutral_and_gross_one():
    sig = _rets([[2.0, 0.0, -1.0]])
    w = strategy.to_weights(sig)
    row = w.iloc[0]
    assert row.sum() == pytest.approx(0.0, abs=1e-12)   # net neutral
    assert row.abs().sum() == pytest.approx(1.0)         # gross 1


def test_to_weights_all_equal_signal_is_flat():
    sig = _rets([[0.5, 0.5, 0.5]])
    w = strategy.to_weights(sig)
    assert w.iloc[0].abs().sum() == pytest.approx(0.0)


def test_momentum_uses_trailing_window_and_is_demeaned():
    # constant positive returns -> all coins equal trailing -> demeaned ~ 0
    r = pd.DataFrame(np.full((30, 3), 0.01), columns=["A", "B", "C"])
    sig = strategy.time_series_momentum_signal(r, lookback=10, skip=1)
    assert sig.iloc[-1].abs().sum() == pytest.approx(0.0, abs=1e-9)


def test_volatility_scale_downweights_high_vol_coin():
    rng = np.random.default_rng(0)
    lo = rng.normal(0, 0.001, 100)
    hi = rng.normal(0, 0.05, 100)
    r = pd.DataFrame({"A": lo, "B": hi})
    sig = pd.DataFrame({"A": np.ones(100), "B": np.ones(100)})
    scaled = strategy.volatility_scale(sig, r, lookback=20)
    # low-vol coin A gets a larger scaled signal than high-vol coin B
    assert abs(scaled["A"].iloc[-1]) > abs(scaled["B"].iloc[-1])


def test_no_trade_band_suppresses_small_moves():
    w = pd.DataFrame({"A": [0.5, 0.505, 0.7], "B": [-0.5, -0.505, -0.7]})
    held = strategy.no_trade_band(w, band=0.02)
    # row1 move 0.005 < band -> stays at row0 values
    assert held.iloc[1]["A"] == pytest.approx(0.5)
    # row2 move 0.195 > band -> updates
    assert held.iloc[2]["A"] == pytest.approx(0.7)


def test_combine_inverse_vol_overweights_calmer_sleeve():
    n = 400
    rng = np.random.default_rng(1)
    calm = pd.Series(rng.normal(0.001, 0.001, n))   # low vol, positive
    wild = pd.Series(rng.normal(0.001, 0.05, n))    # high vol
    combined = strategy.combine_inverse_vol(calm, wild, lookback=30)
    # combined should track the calm sleeve much more closely
    assert combined.corr(calm) > combined.corr(wild)
