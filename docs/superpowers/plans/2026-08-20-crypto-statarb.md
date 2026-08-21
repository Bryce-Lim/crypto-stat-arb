# Crypto Statistical Arbitrage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible crypto stat-arb backtest that combines a cross-sectional reversal sleeve and a time-series momentum sleeve, nets out realistic execution costs, validates out-of-sample, and reports Sharpe / drawdown / alpha-beta.

**Architecture:** Pure, unit-tested library modules (`metrics`, `backtest`, `strategy`, `data`) built bottom-up so all trading logic is testable on synthetic data with no network. A thin `run.py` orchestrator fetches live Binance data, runs the pipeline with a 70/30 train/test split, and emits a metrics table, equity-curve plot, and README numbers. DataFrames are the common currency: rows = UTC dates, columns = coin symbols.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, requests. Testing with pytest.

## Global Constraints

- Frequency: daily. Periods per year for annualization = **365** (crypto trades every day).
- Execution costs: **20 bps** per unit turnover for the market-order case, **7 bps** for the limit-order case. Cost is applied as `turnover * cost_bps / 1e4`.
- No lookahead: weights formed from data through day `t` are earned on day `t+1`. The orchestrator applies `weights.shift(1)` before multiplying by returns.
- Backtest style is **unconstrained**: dollar-neutral (net = 0), gross exposure normalized to 1, no per-position caps.
- Data source: Binance public REST (`https://api.binance.com`), no API key.
- Only stdlib + pandas/numpy/matplotlib/requests. No ML libraries, no ccxt.
- All headline metrics reported **net of costs** and on the **held-out test window**.

---

### Task 0: Project scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Initialize git and package dirs**

```bash
cd C:/Users/Bryce/projects/quant
git init
mkdir -p src tests cache
```

- [ ] **Step 2: Write `requirements.txt`**

```
pandas
numpy
matplotlib
requests
pytest
pyarrow
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
cache/
*.png
.venv/
```

- [ ] **Step 4: Create empty package markers**

Create `src/__init__.py` and `tests/__init__.py` as empty files.

- [ ] **Step 5: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: installs succeed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore src/__init__.py tests/__init__.py
git commit -m "chore: project scaffold"
```

---

### Task 1: Performance metrics

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `annualized_return(returns: pd.Series, periods_per_year: int = 365) -> float`
  - `annualized_vol(returns: pd.Series, periods_per_year: int = 365) -> float`
  - `sharpe_ratio(returns: pd.Series, periods_per_year: int = 365) -> float`
  - `max_drawdown(returns: pd.Series) -> float` (negative fraction)
  - `alpha_beta(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 365) -> tuple[float, float]` (annualized alpha, beta)
  - `summary(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 365) -> dict`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError` (no `src.metrics`).

- [ ] **Step 3: Write the implementation**

```python
# src/metrics.py
import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 365) -> float:
    return float(returns.mean() * periods_per_year)


def annualized_vol(returns: pd.Series, periods_per_year: int = 365) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 365) -> float:
    vol = returns.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(returns.mean() / vol * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def alpha_beta(returns: pd.Series, benchmark: pd.Series,
               periods_per_year: int = 365) -> tuple[float, float]:
    df = pd.concat([returns.rename("y"), benchmark.rename("x")], axis=1).dropna()
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha_daily, beta = float(coef[0]), float(coef[1])
    return alpha_daily * periods_per_year, beta


def summary(returns: pd.Series, benchmark: pd.Series,
            periods_per_year: int = 365) -> dict:
    alpha, beta = alpha_beta(returns, benchmark, periods_per_year)
    return {
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_vol": annualized_vol(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "alpha": alpha,
        "beta": beta,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: performance metrics (sharpe, drawdown, alpha/beta)"
```

---

### Task 2: Unconstrained backtest engine

**Files:**
- Create: `src/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `compute_turnover(weights: pd.DataFrame) -> pd.Series` (row 0 = initial build from zero = sum of abs first-row weights)
  - `run_backtest(weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 20.0) -> pd.Series` — `weights.loc[t]` are the positions *held during* day `t` (caller has already applied any shift); returns net daily strategy return series.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backtest.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest.py -v`
Expected: FAIL (no `src.backtest`).

- [ ] **Step 3: Write the implementation**

```python
# src/backtest.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/backtest.py tests/test_backtest.py
git commit -m "feat: unconstrained backtest engine with turnover costs"
```

---

### Task 3: Signals, weighting, and enhancers

**Files:**
- Create: `src/strategy.py`
- Test: `tests/test_strategy.py`

**Interfaces:**
- Consumes: nothing (operates on price/return/volume DataFrames).
- Produces:
  - `cross_sectional_reversal_signal(returns: pd.DataFrame) -> pd.DataFrame`
  - `time_series_momentum_signal(returns: pd.DataFrame, lookback: int = 20, skip: int = 1) -> pd.DataFrame`
  - `to_weights(signal: pd.DataFrame) -> pd.DataFrame` (per row: demeaned, gross = 1, net = 0)
  - `volatility_scale(signal: pd.DataFrame, returns: pd.DataFrame, lookback: int = 20) -> pd.DataFrame`
  - `volume_filter(signal: pd.DataFrame, volume: pd.DataFrame, lookback: int = 20, cap: float = 3.0) -> pd.DataFrame`
  - `no_trade_band(weights: pd.DataFrame, band: float = 0.02) -> pd.DataFrame`
  - `combine_inverse_vol(ret_a: pd.Series, ret_b: pd.Series, lookback: int = 30) -> pd.Series`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_strategy.py
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


def test_combine_inverse_vol_downweights_wild_sleeve():
    n = 400
    rng = np.random.default_rng(1)
    calm = pd.Series(rng.normal(0.001, 0.001, n))   # low vol
    wild = pd.Series(rng.normal(0.001, 0.05, n))    # high vol
    combined = strategy.combine_inverse_vol(calm, wild, lookback=30)
    equal = 0.5 * calm + 0.5 * wild
    idx = combined.dropna().index
    # inverse-vol weighting down-weights the high-vol sleeve, so the blend
    # is far less volatile than a naive equal-weight blend of the same sleeves.
    # (A correlation test would be wrong here: inverse-vol equalizes risk
    # contributions, so the blend correlates ~equally with both sleeves.)
    assert combined.loc[idx].std() < equal.loc[idx].std()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_strategy.py -v`
Expected: FAIL (no `src.strategy`).

- [ ] **Step 3: Write the implementation**

```python
# src/strategy.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_strategy.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/strategy.py tests/test_strategy.py
git commit -m "feat: reversal/momentum signals, weighting, enhancers"
```

---

### Task 4: Data fetching and universe selection

**Files:**
- Create: `src/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `parse_klines(raw: list) -> pd.DataFrame` — pure; index = UTC date (normalized), columns `close`, `quote_volume`.
  - `fetch_klines(symbol: str, interval: str = "1d", limit: int = 1000, start_ms: int | None = None) -> list` — network.
  - `select_universe(n: int = 30, exclude: set[str] | None = None) -> list[str]` — network; top-`n` USDT symbols by 24h quote volume, stablecoins excluded.
  - `load_panel(symbols: list[str], cache_dir: str = "cache", limit: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]` — returns `(close_df, volume_df)` aligned on a common date index; caches each symbol to parquet.
  - `to_returns(close_df: pd.DataFrame) -> pd.DataFrame`
  - `STABLECOINS: set[str]` constant.

- [ ] **Step 1: Write the failing tests (pure functions + mocked network)**

```python
# tests/test_data.py
import pandas as pd
import pytest
from src import data


def _raw_row(open_ms, close_px, quote_vol):
    # Binance kline: [openTime, open, high, low, close, volume, closeTime,
    #                 quoteAssetVolume, trades, ...]
    return [open_ms, "0", "0", "0", str(close_px), "0",
            open_ms + 1, str(quote_vol), 10, "0", "0", "0"]


def test_parse_klines_extracts_close_and_quote_volume():
    raw = [
        _raw_row(1_600_000_000_000, 100.0, 5000.0),
        _raw_row(1_600_086_400_000, 110.0, 6000.0),
    ]
    df = data.parse_klines(raw)
    assert list(df.columns) == ["close", "quote_volume"]
    assert df["close"].tolist() == [100.0, 110.0]
    assert df["quote_volume"].tolist() == [5000.0, 6000.0]
    assert df.index.is_monotonic_increasing


def test_to_returns_simple_pct_change():
    close = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    rets = data.to_returns(close)
    assert rets["A"].tolist() == pytest.approx([0.1, -0.1])


def test_stablecoins_excluded_by_default_set():
    assert "USDCUSDT" in data.STABLECOINS
    assert "BTCUSDT" not in data.STABLECOINS


def test_select_universe_filters_and_sorts(monkeypatch):
    fake_ticker = [
        {"symbol": "BTCUSDT", "quoteVolume": "100"},
        {"symbol": "ETHUSDT", "quoteVolume": "80"},
        {"symbol": "USDCUSDT", "quoteVolume": "999"},   # stablecoin, drop
        {"symbol": "ETHBTC", "quoteVolume": "70"},       # not USDT, drop
        {"symbol": "SOLUSDT", "quoteVolume": "60"},
    ]
    monkeypatch.setattr(data, "_fetch_24h_tickers", lambda: fake_ticker)
    uni = data.select_universe(n=2)
    assert uni == ["BTCUSDT", "ETHUSDT"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py -v`
Expected: FAIL (no `src.data`).

- [ ] **Step 3: Write the implementation**

```python
# src/data.py
import os
import time
import pandas as pd
import requests

# Binance.US endpoint (api.binance.com returns HTTP 451 from US networks).
# Identical REST shape (/api/v3/klines, /api/v3/ticker/24hr, quoteVolume field).
BASE = "https://api.binance.us"

STABLECOINS = {
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "USDPUSDT", "FDUSDUSDT",
    "DAIUSDT", "USDDUSDT", "EURUSDT", "GBPUSDT", "AEURUSDT",
}


def parse_klines(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["close", "quote_volume"])
    df = pd.DataFrame(raw)
    out = pd.DataFrame({
        "close": df[4].astype(float).values,
        "quote_volume": df[7].astype(float).values,
    }, index=pd.to_datetime(df[0].astype("int64"), unit="ms").dt.normalize())
    out.index.name = "date"
    return out.sort_index()


def fetch_klines(symbol: str, interval: str = "1d", limit: int = 1000,
                 start_ms: int | None = None) -> list:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    r = requests.get(BASE + "/api/v3/klines", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _fetch_24h_tickers() -> list:
    r = requests.get(BASE + "/api/v3/ticker/24hr", timeout=30)
    r.raise_for_status()
    return r.json()


def select_universe(n: int = 30, exclude: set[str] | None = None) -> list[str]:
    exclude = STABLECOINS if exclude is None else exclude
    tickers = _fetch_24h_tickers()
    usdt = [
        t for t in tickers
        if t["symbol"].endswith("USDT") and t["symbol"] not in exclude
        and not t["symbol"].endswith("UPUSDT")
        and not t["symbol"].endswith("DOWNUSDT")
    ]
    usdt.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
    return [t["symbol"] for t in usdt[:n]]


def load_panel(symbols: list[str], cache_dir: str = "cache",
               limit: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(cache_dir, exist_ok=True)
    closes, vols = {}, {}
    for sym in symbols:
        path = os.path.join(cache_dir, f"{sym}_1d.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
        else:
            df = parse_klines(fetch_klines(sym, limit=limit))
            if not df.empty:
                df.to_parquet(path)
            time.sleep(0.2)  # be polite to the API
        if df.empty:
            continue
        closes[sym] = df["close"]
        vols[sym] = df["quote_volume"]
    close_df = pd.DataFrame(closes).sort_index()
    vol_df = pd.DataFrame(vols).sort_index()
    # keep only fully-populated dates to avoid fabricated cross-section rows
    close_df = close_df.dropna(how="any")
    vol_df = vol_df.reindex(close_df.index)
    return close_df, vol_df


def to_returns(close_df: pd.DataFrame) -> pd.DataFrame:
    return close_df.pct_change().dropna(how="all")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Live smoke test (network; run manually)**

Run:
```bash
python -c "from src import data; u=data.select_universe(5); print(u); c,v=data.load_panel(u, limit=100); print(c.shape); print(c.tail(2))"
```
Expected: prints 5 symbols, a `(rows, 5)` shape, and recent close prices. If the network is unavailable, note it and proceed — unit tests already cover the logic.

- [ ] **Step 6: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: Binance data fetch, universe selection, caching"
```

---

### Task 5: Orchestration, OOS split, and reporting

**Files:**
- Create: `src/pipeline.py`
- Create: `run.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `src.data`, `src.strategy`, `src.backtest`, `src.metrics`.
- Produces:
  - `reversal_returns(returns, volume, cost_bps=20.0, vol_lookback=20, vol_z_cap=3.0, band=0.02) -> pd.Series` — full enhanced reversal sleeve, lookahead-safe (applies `weights.shift(1)`).
  - `momentum_returns(returns, cost_bps=20.0, lookback=20, band=0.02) -> pd.Series`
  - `split_index(index, train_frac=0.7) -> tuple[pd.Index, pd.Index]`
  - `select_momentum_lookback(returns, volume, grid, cost_bps=20.0) -> int` — picks the lookback with best **train** combined Sharpe.
  - `build_combined(returns, volume, mom_lookback, cost_bps=20.0) -> tuple[pd.Series, pd.Series, pd.Series]` — `(reversal, momentum, combined)`.

- [ ] **Step 1: Write the failing tests (synthetic, no network)**

```python
# tests/test_pipeline.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL (no `src.pipeline`).

- [ ] **Step 3: Write the implementation**

```python
# src/pipeline.py
import pandas as pd
from src import strategy, backtest, metrics


def reversal_returns(returns: pd.DataFrame, volume: pd.DataFrame,
                     cost_bps: float = 20.0, vol_lookback: int = 20,
                     vol_z_cap: float = 3.0, band: float = 0.02) -> pd.Series:
    sig = strategy.cross_sectional_reversal_signal(returns)
    sig = strategy.volatility_scale(sig, returns, lookback=vol_lookback)
    sig = strategy.volume_filter(sig, volume, lookback=vol_lookback,
                                 cap=vol_z_cap)
    weights = strategy.to_weights(sig.fillna(0.0))
    weights = strategy.no_trade_band(weights, band=band)
    held = weights.shift(1).fillna(0.0)         # no lookahead
    return backtest.run_backtest(held, returns, cost_bps=cost_bps)


def momentum_returns(returns: pd.DataFrame, cost_bps: float = 20.0,
                     lookback: int = 20, band: float = 0.02) -> pd.Series:
    sig = strategy.time_series_momentum_signal(returns, lookback=lookback)
    weights = strategy.to_weights(sig.fillna(0.0))
    weights = strategy.no_trade_band(weights, band=band)
    held = weights.shift(1).fillna(0.0)
    return backtest.run_backtest(held, returns, cost_bps=cost_bps)


def split_index(index, train_frac: float = 0.7):
    n = int(len(index) * train_frac)
    return index[:n], index[n:]


def build_combined(returns: pd.DataFrame, volume: pd.DataFrame,
                   mom_lookback: int, cost_bps: float = 20.0):
    rev = reversal_returns(returns, volume, cost_bps=cost_bps)
    mom = momentum_returns(returns, cost_bps=cost_bps, lookback=mom_lookback)
    comb = strategy.combine_inverse_vol(rev.fillna(0.0), mom.fillna(0.0))
    return rev, mom, comb


def select_momentum_lookback(returns: pd.DataFrame, volume: pd.DataFrame,
                             grid: list[int], cost_bps: float = 20.0) -> int:
    train_idx, _ = split_index(returns.index)
    best_lb, best_sharpe = grid[0], -1e9
    for lb in grid:
        _, _, comb = build_combined(returns, volume, mom_lookback=lb,
                                    cost_bps=cost_bps)
        s = metrics.sharpe_ratio(comb.loc[train_idx].dropna())
        if s > best_sharpe:
            best_sharpe, best_lb = s, lb
    return best_lb
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Write `run.py` (the end-to-end orchestrator)**

```python
# run.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import data, pipeline, metrics


def _fmt(d: dict) -> str:
    return (f"ann_return={d['ann_return']:.2%}  ann_vol={d['ann_vol']:.2%}  "
            f"sharpe={d['sharpe']:.2f}  maxDD={d['max_drawdown']:.2%}  "
            f"alpha={d['alpha']:.2%}  beta={d['beta']:.2f}")


def main():
    print("Selecting universe...")
    universe = data.select_universe(n=30)
    print("Loading price/volume panel...")
    close, volume = data.load_panel(universe)
    returns = data.to_returns(close)
    volume = volume.reindex(returns.index)

    # BTC benchmark for alpha/beta
    btc = returns["BTCUSDT"] if "BTCUSDT" in returns else returns.mean(axis=1)

    # ---- Out-of-sample: pick momentum lookback on TRAIN only ----
    grid = [10, 20, 30, 40]
    mom_lb = pipeline.select_momentum_lookback(returns, volume, grid)
    print(f"Selected momentum lookback (train): {mom_lb}")

    rev, mom, comb = pipeline.build_combined(returns, volume, mom_lb)
    train_idx, test_idx = pipeline.split_index(returns.index)

    stages = {
        "Reversal (net)": rev,
        "Momentum (net)": mom,
        "Combined (net)": comb,
    }

    print("\n=== TRAIN (in-sample) ===")
    for name, series in stages.items():
        s = series.loc[train_idx].dropna()
        print(f"{name:20s} {_fmt(metrics.summary(s, btc.loc[s.index]))}")

    print("\n=== TEST (out-of-sample, headline) ===")
    test_rows = {}
    for name, series in stages.items():
        s = series.loc[test_idx].dropna()
        d = metrics.summary(s, btc.loc[s.index])
        test_rows[name] = d
        print(f"{name:20s} {_fmt(d)}")

    # ---- Equity curve on the full sample ----
    plt.figure(figsize=(10, 6))
    for name, series in stages.items():
        eq = (1.0 + series.fillna(0.0)).cumprod()
        plt.plot(eq.index, eq.values, label=name)
    plt.axvline(test_idx.min(), color="k", linestyle="--", alpha=0.5,
                label="train/test split")
    plt.legend()
    plt.title("Crypto Stat-Arb: Cumulative Growth (net of costs)")
    plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=120)
    print("\nSaved equity_curve.png")

    pd.DataFrame(test_rows).T.to_csv("test_metrics.csv")
    print("Saved test_metrics.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the full pipeline (network)**

Run: `python run.py`
Expected: prints selected lookback, TRAIN and TEST metric tables, and writes `equity_curve.png` + `test_metrics.csv`. Record the TEST combined Sharpe / max drawdown / alpha / beta — these are the headline resume numbers. If the network is unavailable, note it; all logic is already unit-tested.

- [ ] **Step 7: Commit**

```bash
git add src/pipeline.py run.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestration, OOS split, reporting"
```

---

### Task 6: README writeup

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: the printed TEST metrics and `equity_curve.png` from Task 5, Step 6.

- [ ] **Step 1: Write `README.md`** (fill the bracketed numbers from the Task 5 run)

```markdown
# Crypto Statistical Arbitrage

A dollar-neutral, market-neutral statistical-arbitrage strategy on the ~30 most
liquid Binance USDT pairs, combining a **cross-sectional reversal** sleeve and a
**time-series momentum** sleeve via inverse-volatility weighting. All results are
**net of execution costs** (20 bps market-order assumption) and reported on a
**held-out out-of-sample** window.

## Headline results (out-of-sample, net of costs)

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Alpha vs BTC | Beta vs BTC |
|---|---|---|---|---|---|---|
| Reversal | [..] | [..] | [..] | [..] | [..] | [..] |
| Momentum | [..] | [..] | [..] | [..] | [..] | [..] |
| **Combined** | [..] | [..] | **[..]** | [..] | [..] | [..] |

![equity curve](equity_curve.png)

## Method

- **Universe:** top ~30 USDT pairs by 24h volume, stablecoins excluded.
- **Reversal signal:** short the day's relative winners, long the relative losers
  (cross-sectionally demeaned prior-day return).
- **Enhancers:** inverse-volatility position scaling; a volume-spike filter that
  amplifies reversal on liquidation-driven moves (uninformed trading); and a
  no-trade band that cuts turnover to control the 20 bps cost drag.
- **Momentum sleeve:** demeaned trailing return, lookback chosen on the training
  window only.
- **Weighting:** the two sleeves are blended by inverse realized volatility
  (risk parity), which raises the combined Sharpe through diversification.
- **Validation:** hyperparameters fit on the first 70% of history; metrics above
  are the untouched final 30%.

## Run it

    pip install -r requirements.txt
    python run.py          # fetches data, backtests, writes metrics + plot
    pytest -q              # unit tests for signals, backtest, metrics

## Notes / honesty

Universe is current-liquid coins, so there is mild survivorship bias (stated, not
hidden). Costs modeled as `turnover x cost_bps`; the 7 bps limit-order case is a
one-line change to `cost_bps`.
```

- [ ] **Step 2: Verify the full test suite is green**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: resume-facing README with headline numbers"
```

---

## Notes for the implementer

- Run tasks in order; each builds on the previous module's public functions.
- Network is needed only for the live smoke test (Task 4, Step 5) and the full run
  (Task 5, Step 6). All logic is unit-tested offline, so a firewalled environment
  can still complete Tasks 0–6 except those two steps — note them as skipped.
- After Task 5's live run, paste the real TEST numbers into the README table in Task 6.
- Keep `cost_bps=20.0` as the headline; mention the 7 bps limit-order variant in prose.
