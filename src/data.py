import os
import time
import pandas as pd
import requests

BASE = "https://api.binance.com"

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
