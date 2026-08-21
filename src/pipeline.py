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
