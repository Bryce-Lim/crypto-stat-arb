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
