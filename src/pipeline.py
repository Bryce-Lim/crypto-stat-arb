import pandas as pd
from src import strategy, backtest, metrics


def split_index(index, train_frac: float = 0.7):
    n = int(len(index) * train_frac)
    return index[:n], index[n:]


def reversal_returns(returns: pd.DataFrame, cost_bps: float = 7.0,
                     band: float = 0.05,
                     target_vol: float | None = 0.15) -> pd.Series:
    """Final book: cross-sectional daily reversal, normalized to a dollar-neutral
    gross-1 portfolio, with a no-trade band for turnover/cost control and optional
    book-level volatility targeting. No lookahead: weights are shifted one day
    before earning returns."""
    sig = strategy.cross_sectional_reversal_signal(returns)
    weights = strategy.to_weights(sig.fillna(0.0))
    weights = strategy.no_trade_band(weights, band=band)
    held = weights.shift(1).fillna(0.0)
    r = backtest.run_backtest(held, returns, cost_bps=cost_bps)
    if target_vol is not None:
        r = strategy.vol_target(r, target_annual=target_vol)
    return r


def momentum_returns(returns: pd.DataFrame, cost_bps: float = 7.0,
                     lookback: int = 60,
                     target_vol: float | None = 0.15) -> pd.Series:
    """Time-series momentum sleeve. Retained for the out-of-sample ablation
    (it does not generalize OOS), not part of the final book."""
    sig = strategy.time_series_momentum_signal(returns, lookback=lookback)
    weights = strategy.to_weights(sig.fillna(0.0))
    held = weights.shift(1).fillna(0.0)
    r = backtest.run_backtest(held, returns, cost_bps=cost_bps)
    if target_vol is not None:
        r = strategy.vol_target(r, target_annual=target_vol)
    return r


def select_band(returns: pd.DataFrame, grid: list[float],
                cost_bps: float = 7.0) -> float:
    """Pick the no-trade band that maximizes reversal Sharpe on the TRAIN slice
    only (out-of-sample discipline: the test slice is never touched here)."""
    train_idx, _ = split_index(returns.index)
    best, best_sharpe = grid[0], -1e9
    for b in grid:
        r = reversal_returns(returns, cost_bps=cost_bps, band=b, target_vol=None)
        s = metrics.sharpe_ratio(r.loc[train_idx].dropna())
        if s > best_sharpe:
            best_sharpe, best = s, b
    return best
