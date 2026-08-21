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
