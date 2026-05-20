from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_dn = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append normalized technical features used by the RL observation."""
    out = df.copy()
    close = out["close"]

    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_15"] = close.pct_change(15)

    ema_fast = _ema(close, 9)
    ema_slow = _ema(close, 21)
    out["ema_diff"] = (ema_fast - ema_slow) / close

    macd = _ema(close, 12) - _ema(close, 26)
    macd_sig = _ema(macd, 9)
    out["macd"] = macd / close
    out["macd_sig"] = macd_sig / close
    out["macd_hist"] = (macd - macd_sig) / close

    out["rsi"] = (_rsi(close, 14) - 50.0) / 50.0

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=0)
    out["bb_pos"] = (close - sma20) / (2 * std20.replace(0, np.nan))

    atr = _atr(out, 14)
    out["atr_pct"] = atr / close

    vol_ma = out["volume"].rolling(20).mean()
    out["vol_ratio"] = (out["volume"] / vol_ma.replace(0, np.nan)).clip(0, 10) / 10.0

    out["hl_range"] = (out["high"] - out["low"]) / close

    feature_cols = [
        "ret_1", "ret_5", "ret_15",
        "ema_diff", "macd", "macd_sig", "macd_hist",
        "rsi", "bb_pos", "atr_pct", "vol_ratio", "hl_range",
    ]
    out = out.replace([np.inf, -np.inf], np.nan)
    out[feature_cols] = out[feature_cols].astype(np.float32)
    out.attrs["feature_cols"] = feature_cols
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return list(df.attrs.get("feature_cols", []))
