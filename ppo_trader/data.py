from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from ppo_trader.config import DataConfig
from ppo_trader.features import add_features


def _cache_path(cache_dir: str, symbol: str, interval: str, period: str) -> Path:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Path(cache_dir) / f"{symbol}_{interval}_{period}.parquet"


def fetch_symbol(
    symbol: str,
    interval: str = "5m",
    period: str = "60d",
    cache_dir: Optional[str] = "data_cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV for a single symbol from Yahoo Finance with optional disk cache."""
    if cache_dir and use_cache:
        p = _cache_path(cache_dir, symbol, interval, period)
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
        df.to_parquet(_cache_path(cache_dir, symbol, interval, period))
    return df


def load_dataset(
    symbols: List[str],
    cfg: DataConfig = DataConfig(),
) -> Dict[str, pd.DataFrame]:
    """Fetch + feature-engineer all symbols. Returns dict symbol -> DataFrame."""
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
