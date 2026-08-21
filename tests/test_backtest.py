import numpy as np
import pandas as pd
import pytest
from src import backtest


def _frame(vals):
    idx = pd.RangeIndex(len(vals))
    return pd.DataFrame(vals, index=idx, columns=["A", "B"])


def test_turnover_first_row_is_initial_build():
    w = _frame([[0.5, -0.5], [0.5, -0.5]])
    to = backtest.compute_turnover(w)
    assert to.iloc[0] == pytest.approx(1.0)  # built 0.5 + 0.5 from zero
    assert to.iloc[1] == pytest.approx(0.0)  # unchanged


def test_turnover_on_flip():
    w = _frame([[0.5, -0.5], [-0.5, 0.5]])
    to = backtest.compute_turnover(w)
    assert to.iloc[1] == pytest.approx(2.0)  # |−1| + |1|


def test_gross_return_no_cost():
    w = _frame([[0.5, -0.5]])
    r = _frame([[0.04, 0.02]])  # 0.5*0.04 + (-0.5)*0.02 = 0.01
    net = backtest.run_backtest(w, r, cost_bps=0.0)
    assert net.iloc[0] == pytest.approx(0.01)


def test_cost_reduces_return():
    w = _frame([[0.5, -0.5]])
    r = _frame([[0.0, 0.0]])
    net = backtest.run_backtest(w, r, cost_bps=20.0)
    # turnover 1.0 * 20bps = 0.002 drag
    assert net.iloc[0] == pytest.approx(-0.002)
