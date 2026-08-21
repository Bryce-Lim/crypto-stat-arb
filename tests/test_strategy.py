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


def test_no_trade_band_holds_through_all_nan_row():
    w = pd.DataFrame({"A": [0.5, np.nan, 0.5], "B": [-0.5, np.nan, -0.5]})
    held = strategy.no_trade_band(w, band=0.02)
    # a mid-series all-NaN row (data gap) must carry the prior book forward,
    # not reset to flat (which would book two spurious round-trip trades)
    assert held.iloc[1]["A"] == pytest.approx(0.5)
    assert held.iloc[1]["B"] == pytest.approx(-0.5)


def test_vol_target_scales_toward_target_annual_vol():
    rng = np.random.default_rng(0)
    # ~76% annualized vol input, well above the 15% target
    r = pd.Series(rng.normal(0.0, 0.04, 2000))
    out = strategy.vol_target(r, target_annual=0.15, lookback=30).dropna()
    ann = out.std(ddof=0) * np.sqrt(365)
    assert 0.10 < ann < 0.22        # pulled close to 15%


def test_vol_target_keeps_a_profitable_series_profitable():
    # under time-varying vol the scaler re-weights days, so Sharpe shifts a bit,
    # but a clearly profitable series must stay profitable and same-order (the
    # transform sets a risk budget, it does not create or destroy the edge)
    rng = np.random.default_rng(1)
    n = 3000
    vol = np.repeat(rng.uniform(0.01, 0.05, 60), 50)[:n]  # regime-switching vol
    r = pd.Series(0.002 + rng.normal(0, 1, n) * vol)      # positive daily drift
    out = strategy.vol_target(r, target_annual=0.15, lookback=30)
    idx = out.dropna().index
    sh_in = r.loc[idx].mean() / r.loc[idx].std(ddof=0)
    sh_out = out.loc[idx].mean() / out.loc[idx].std(ddof=0)
    assert sh_in > 0 and sh_out > 0
    assert 0.3 < sh_out / sh_in < 3.0
