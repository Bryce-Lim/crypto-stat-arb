"""End-to-end crypto statistical-arbitrage backtest.

Builds an established, liquid crypto universe, runs a cross-sectional daily
reversal strategy net of execution costs, validates it out-of-sample, and shows
the enhancer ablation (including a momentum sleeve that does NOT generalize).

Run: python run.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data, strategy, backtest, metrics, pipeline

TARGET_VOL = 0.15
BAND_GRID = [0.0, 0.02, 0.05, 0.10, 0.20]


def _net(weights, returns, cost_bps):
    return backtest.run_backtest(weights.shift(1).fillna(0.0), returns, cost_bps)


def _sharpe_net(weights, returns, cost_bps, idx=None):
    r = _net(weights, returns, cost_bps)
    r = r.loc[idx] if idx is not None else r
    return metrics.sharpe_ratio(r.dropna())


def _fmt(d):
    return (f"sharpe={d['sharpe']:+.2f}  annRet={d['ann_return']:+.1%}  "
            f"annVol={d['ann_vol']:.1%}  maxDD={d['max_drawdown']:.1%}  "
            f"alpha={d['alpha']:+.1%}  beta={d['beta']:+.2f}")


def main():
    print("Building established, liquid universe (this fetches/caches data)...")
    universe, close, volume = data.build_universe(n=30, pool=80, min_history=700)
    returns = data.to_returns(close)
    volume = volume.reindex(returns.index)
    btc = returns["BTCUSDT"] if "BTCUSDT" in returns else returns.mean(axis=1)
    print(f"Universe: {len(universe)} coins | Panel: {len(returns)} days "
          f"{returns.index.min().date()} -> {returns.index.max().date()}")

    # ---- OOS discipline: choose the no-trade band on TRAIN only ----
    band = pipeline.select_band(returns, BAND_GRID, cost_bps=7.0)
    train_idx, test_idx = pipeline.split_index(returns.index)
    print(f"Selected no-trade band (train-only): {band}")

    # ---- Enhancer ablation (full-sample net Sharpe at both cost levels) ----
    rev_sig = strategy.cross_sectional_reversal_signal(returns)
    w_raw = strategy.to_weights(rev_sig.fillna(0.0))
    w_band = strategy.no_trade_band(w_raw, band=band)
    w_vs = strategy.no_trade_band(
        strategy.to_weights(strategy.volatility_scale(rev_sig, returns).fillna(0.0)),
        band=band)
    w_vf = strategy.no_trade_band(
        strategy.to_weights(strategy.volume_filter(rev_sig, volume).fillna(0.0)),
        band=band)
    w_mom = strategy.to_weights(
        strategy.time_series_momentum_signal(returns, lookback=60).fillna(0.0))

    print("\n=== ENHANCER ABLATION (full-sample net Sharpe) ===")
    print(f"{'variant':32s} {'7bps':>8s} {'20bps':>8s}")
    for label, w in [
        ("reversal (raw)", w_raw),
        (f"reversal + no-trade band({band})", w_band),
        ("reversal + band + vol-scaling", w_vs),
        ("reversal + band + volume filter", w_vf),
        ("momentum sleeve (lb=60)", w_mom),
    ]:
        print(f"{label:32s} {_sharpe_net(w, returns, 7.0):>+8.2f} "
              f"{_sharpe_net(w, returns, 20.0):>+8.2f}")
    print("(vol-scaling and the volume filter do not improve net returns -> "
          "final book uses reversal + no-trade band only)")

    # ---- Momentum fails out-of-sample (the reason it is excluded) ----
    mom = pipeline.momentum_returns(returns, cost_bps=7.0, lookback=60)
    print("\n=== MOMENTUM SLEEVE: in- vs out-of-sample (7bps) ===")
    print(f"  TRAIN sharpe={metrics.sharpe_ratio(mom.loc[train_idx].dropna()):+.2f}"
          f"   TEST sharpe={metrics.sharpe_ratio(mom.loc[test_idx].dropna()):+.2f}"
          f"   -> does not generalize, dropped from final book")

    # ---- FINAL STRATEGY: reversal + band + vol-target, at both cost levels ----
    print("\n=== FINAL STRATEGY: cross-sectional reversal (vol-targeted 15%) ===")
    headline = {}
    for cost, tag in [(7.0, "7bps limit orders"), (20.0, "20bps market orders")]:
        r = pipeline.reversal_returns(returns, cost_bps=cost, band=band,
                                      target_vol=TARGET_VOL)
        print(f"\n  --- {tag} ---")
        for lbl, idx in [("TRAIN", train_idx), ("TEST ", test_idx),
                         ("FULL ", returns.index)]:
            s = r.loc[idx].dropna()
            d = metrics.summary(s, btc.loc[s.index])
            print(f"    {lbl}: {_fmt(d)}")
            if lbl == "TEST " or lbl == "FULL ":
                headline[f"{tag} | {lbl.strip()}"] = d

    # ---- Equity curve (final strategy, 7bps) vs BTC buy-and-hold ----
    r7 = pipeline.reversal_returns(returns, cost_bps=7.0, band=band,
                                   target_vol=TARGET_VOL).fillna(0.0)
    plt.figure(figsize=(10, 6))
    plt.plot((1 + r7).cumprod().index, (1 + r7).cumprod().values,
             label="Reversal strategy (7bps, vol-targeted)")
    plt.plot((1 + btc).cumprod().index, (1 + btc).cumprod().values,
             label="BTC buy & hold", alpha=0.6)
    plt.axvline(test_idx.min(), color="k", linestyle="--", alpha=0.5,
                label="train/test split")
    plt.legend()
    plt.title("Crypto Cross-Sectional Reversal — Growth of $1 (net of costs)")
    plt.ylabel("Growth of $1")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=120)
    pd.DataFrame(headline).T.to_csv("results.csv")
    print("\nSaved equity_curve.png and results.csv")


if __name__ == "__main__":
    main()
