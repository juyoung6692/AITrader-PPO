from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from ppo_trader.config import EnvConfig
from ppo_trader.env import DayTradingEnv
from ppo_trader.features import add_features


SYMBOL = os.getenv("TRADE_SYMBOL", "AAPL")
MODEL_PATH = os.getenv("MODEL_PATH", "models/ppo_daytrader")
NOTIONAL_USD = float(os.getenv("NOTIONAL_USD", "10000"))
EOD_FLATTEN_MIN = float(os.getenv("EOD_FLATTEN_MIN", "10"))


def _trading_client() -> TradingClient:
    return TradingClient(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_API_SECRET"],
        paper=True,
    )


def _data_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_API_SECRET"],
    )


def fetch_recent_bars(data: StockHistoricalDataClient, symbol: str, lookback_min: int = 600) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_min)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
    )
    bars = data.get_stock_bars(req).df
    if bars is None or bars.empty:
        raise RuntimeError(f"no bars for {symbol}")
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.droplevel(0)
    bars.index = pd.to_datetime(bars.index, utc=True)
    return bars[["open", "high", "low", "close", "volume"]].copy()


def current_position(trade: TradingClient, symbol: str) -> int:
    try:
        pos = trade.get_open_position(symbol)
    except Exception:
        return 0
    qty = float(getattr(pos, "qty", 0) or 0)
    if qty > 0:
        return 1
    if qty < 0:
        return -1
    return 0


def close_position(trade: TradingClient, symbol: str) -> None:
    try:
        trade.close_position(symbol)
    except Exception as exc:
        print(f"[live] close_position failed: {exc}")


def submit_market(trade: TradingClient, symbol: str, side: OrderSide, notional: float) -> None:
    order = MarketOrderRequest(
        symbol=symbol,
        notional=str(round(notional, 2)),
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    trade.submit_order(order)


def market_status(trade: TradingClient) -> tuple[bool, float]:
    clock = trade.get_clock()
    if not clock.is_open:
        return False, -1.0
    mtc = (clock.next_close - clock.timestamp).total_seconds() / 60.0
    return True, mtc


def decide_target(model: PPO, feat: pd.DataFrame, env_cfg: EnvConfig) -> int:
    env = DayTradingEnv(dataset={SYMBOL: feat}, cfg=env_cfg, random_symbol=False, seed=0)
    env.reset()
    env.t = len(feat) - 1
    obs = env._obs()
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    if action == 1:
        return 1
    if action == 2 and env_cfg.allow_short:
        return -1
    return 0


def main() -> int:
    trade = _trading_client()
    is_open, mtc = market_status(trade)
    if not is_open:
        print("[live] market closed; skip")
        return 0

    data = _data_client()
    try:
        bars = fetch_recent_bars(data, SYMBOL)
    except Exception as exc:
        print(f"[live] data fetch failed: {exc}")
        return 0

    feat = add_features(bars).dropna()
    if len(feat) < 35:
        print(f"[live] not enough bars yet ({len(feat)})")
        return 0

    env_cfg = EnvConfig()
    model = PPO.load(MODEL_PATH)
    target = decide_target(model, feat, env_cfg)

    if 0 < mtc <= EOD_FLATTEN_MIN:
        print(f"[live] near close ({mtc:.1f}m) -> force flat")
        target = 0

    pos_now = current_position(trade, SYMBOL)
    print(f"[live] symbol={SYMBOL} target={target} pos={pos_now} mtc={mtc:.1f}")

    if target == pos_now:
        return 0

    if pos_now != 0:
        close_position(trade, SYMBOL)

    if target == 1:
        submit_market(trade, SYMBOL, OrderSide.BUY, NOTIONAL_USD)
    elif target == -1:
        submit_market(trade, SYMBOL, OrderSide.SELL, NOTIONAL_USD)

    return 0


if __name__ == "__main__":
    sys.exit(main())
