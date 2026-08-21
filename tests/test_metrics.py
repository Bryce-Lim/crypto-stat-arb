import numpy as np
import pandas as pd
import pytest
from src import metrics


def test_sharpe_of_constant_positive_returns_is_large():
    r = pd.Series([0.001] * 365)
    # zero vol -> guarded to 0.0
    assert metrics.sharpe_ratio(r) == 0.0


def test_sharpe_scales_with_mean_over_vol():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 3650))
    s = metrics.sharpe_ratio(r)
    expected = (r.mean() / r.std(ddof=0)) * np.sqrt(365)
    assert s == pytest.approx(expected)


def test_annualized_vol():
    r = pd.Series([0.01, -0.01, 0.02, -0.02])
    assert metrics.annualized_vol(r) == pytest.approx(r.std(ddof=0) * np.sqrt(365))


def test_max_drawdown_simple():
    # +10% then -50% -> equity 1.1 then 0.55, peak 1.1 -> dd = 0.55/1.1 - 1 = -0.5
    r = pd.Series([0.10, -0.50])
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_alpha_beta_recovers_known_line():
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0, 0.02, 2000))
    true_beta, true_alpha_daily = 0.5, 0.0003
    strat = true_alpha_daily + true_beta * bench
    alpha_ann, beta = metrics.alpha_beta(strat, bench)
    assert beta == pytest.approx(true_beta, abs=1e-6)
    assert alpha_ann == pytest.approx(true_alpha_daily * 365, abs=1e-6)


def test_summary_keys():
    r = pd.Series(np.random.default_rng(2).normal(0.001, 0.01, 500))
    b = pd.Series(np.random.default_rng(3).normal(0.001, 0.02, 500))
    out = metrics.summary(r, b)
    assert set(out) == {
        "ann_return", "ann_vol", "sharpe", "max_drawdown", "alpha", "beta"
    }


def test_information_coefficient_detects_perfect_and_zero_signal():
    idx = pd.date_range("2020-01-01", periods=60)
    cols = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(0)
    fwd = pd.DataFrame(rng.normal(0, 0.02, (60, 5)), index=idx, columns=cols)
    # signal == forward return  -> IC ~ 1, huge t-stat
    ic, t = metrics.information_coefficient(fwd.copy(), fwd)
    assert ic > 0.99 and t > 10
    # signal independent of forward return -> t-stat near zero
    noise = pd.DataFrame(rng.normal(0, 1, (60, 5)), index=idx, columns=cols)
    ic0, t0 = metrics.information_coefficient(noise, fwd)
    assert abs(t0) < 3
