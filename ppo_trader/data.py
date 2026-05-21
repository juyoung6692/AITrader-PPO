from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from ppo_trader.config import DataConfig
from ppo_trader.features import add_features


_INTERVAL_RE = re.compile(r"^(\d+)\s*(m|min|minute|h|hour|d|day)$", re.IGNORECASE)


def _cache_path(cache_dir: str, symbol: str, interval: str, period: str, source: str) -> Path:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Path(cache_dir) / f"{source}_{symbol}_{interval}_{period}.parquet"


def _period_to_days(period: str) -> int:
    s = period.strip().lower()
    if s.endswith("d"):
        return int(s[:-1])
    if s.endswith("mo"):
        return int(s[:-2]) * 30
    if s.endswith("y"):
        return int(s[:-1]) * 365
    return int(s)


def _interval_to_alpaca(interval: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    m = _INTERVAL_RE.match(interval)
    if not m:
        raise ValueError(f"unsupported interval for alpaca: {interval}")
    n, unit = int(m.group(1)), m.group(2).lower()
    if unit.startswith("m"):
        return TimeFrame(n, TimeFrameUnit.Minute)
    if unit.startswith("h"):
        return TimeFrame(n, TimeFrameUnit.Hour)
    if unit.startswith("d"):
        return TimeFrame(n, TimeFrameUnit.Day)
    raise ValueError(f"unsupported interval for alpaca: {interval}")


def fetch_symbol_yfinance(
    symbol: str,
    interval: str = "5m",
    period: str = "60d",
    cache_dir: Optional[str] = "data_cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    if cache_dir and use_cache:
        p = _cache_path(cache_dir, symbol, interval, period, "yf")
        if p.exists():
            df = pd.read_parquet(p)
            if not df.empty:
                return df

    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} interval={interval} period={period}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)

    if cache_dir:
        df.to_parquet(_cache_path(cache_dir, symbol, interval, period, "yf"))
    return df


def fetch_symbol_alpaca(
    symbol: str,
    interval: str = "5m",
    period: str = "1825d",
    cache_dir: Optional[str] = "data_cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    if cache_dir and use_cache:
        p = _cache_path(cache_dir, symbol, interval, period, "alp")
        if p.exists():
            df = pd.read_parquet(p)
            if not df.empty:
                return df

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.enums import DataFeed

    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_API_KEY/ALPACA_API_SECRET required for DATA_SOURCE=alpaca")

    client = StockHistoricalDataClient(api_key, api_secret)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=_period_to_days(period))

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=_interval_to_alpaca(interval),
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(req).df
    if bars is None or bars.empty:
        raise RuntimeError(f"No alpaca data for {symbol} interval={interval} period={period}")

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.droplevel(0)
    bars.index = pd.to_datetime(bars.index, utc=True)
    df = bars[["open", "high", "low", "close", "volume"]].dropna().copy()

    if cache_dir:
        df.to_parquet(_cache_path(cache_dir, symbol, interval, period, "alp"))
    return df


def fetch_symbol(
    symbol: str,
    interval: str = "5m",
    period: str = "60d",
    cache_dir: Optional[str] = "data_cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    source = os.environ.get("DATA_SOURCE", "yfinance").lower()
    if source == "alpaca":
        return fetch_symbol_alpaca(symbol, interval, period, cache_dir, use_cache)
    return fetch_symbol_yfinance(symbol, interval, period, cache_dir, use_cache)


def load_dataset(
    symbols: List[str],
    cfg: DataConfig = DataConfig(),
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            raw = fetch_symbol(s, cfg.interval, cfg.period, cfg.cache_dir)
        except Exception as e:
            print(f"[data] skip {s}: {e}")
            continue
        feat = add_features(raw).dropna()
        if len(feat) < 100:
            print(f"[data] skip {s}: too few bars ({len(feat)})")
            continue
        out[s] = feat
        print(f"[data] {s}: {len(feat)} bars")
    if not out:
        raise RuntimeError("No usable symbols loaded.")
    return out


def split_train_eval(
    dataset: Dict[str, pd.DataFrame], train_split: float = 0.8
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    tr: Dict[str, pd.DataFrame] = {}
    ev: Dict[str, pd.DataFrame] = {}
    for s, df in dataset.items():
        n = len(df)
        cut = int(n * train_split)
        tr[s] = df.iloc[:cut].copy()
        ev[s] = df.iloc[cut:].copy()
    return tr, ev
