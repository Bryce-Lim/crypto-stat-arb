import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 365) -> float:
    return float(returns.mean() * periods_per_year)


def annualized_vol(returns: pd.Series, periods_per_year: int = 365) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 365) -> float:
    vol = returns.std(ddof=0)
    if np.isclose(vol, 0) or np.isnan(vol):
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
