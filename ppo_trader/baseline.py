"""Non-RL baselines that share the DayTradingEnv backtest harness.

Lets you compare the PPO policy against trivial strategies (buy & hold,
RSI mean-reversion). Output uses the same table format as evaluate.py so
results stack side by side.
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd

from ppo_trader.config import DEFAULT_SYMBOLS, DataConfig, EnvConfig
from ppo_trader.data import load_dataset, split_train_eval
from ppo_trader.env import DayTradingEnv
from ppo_trader.evaluate import _metrics, _print_table


class AlwaysLongPolicy:
    """Buy at the first step, hold until end-of-data (modulo EOD flat rule)."""

    name = "always_long"

    def act(self, env: DayTradingEnv) -> int:
        return 1


class RsiMeanReversionPolicy:
    """Long when RSI is oversold, flat when overbought."""

    name = "rsi_meanrev"

    def __init__(self, low: float = -0.4, high: float = 0.4) -> None:
        self.low = low
        self.high = high

    def act(self, env: DayTradingEnv) -> int:
        rsi_norm = float(env.df["rsi"].iat[env.t])
        if rsi_norm <= self.low:
            return 1
        if rsi_norm >= self.high:
            return 0
        return 1 if env.position == 1 else 0


POLICIES = {
    "always_long": AlwaysLongPolicy,
    "rsi_meanrev": RsiMeanReversionPolicy,
}


def run_backtest(
    policy,
    dataset: Dict[str, pd.DataFrame],
    env_cfg: EnvConfig,
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for symbol, df in dataset.items():
        env = DayTradingEnv(
            dataset={symbol: df}, cfg=env_cfg, random_symbol=False, seed=0
        )
        env.reset()
        equity_curve: List[float] = [env.equity]
        done = False
        while not done:
            action = policy.act(env)
            _, _, terminated, truncated, _ = env.step(int(action))
            equity_curve.append(env.equity)
            done = terminated or truncated
        summary[symbol] = _metrics(equity_curve, env_cfg.initial_cash)
        summary[symbol]["bars"] = len(df)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run rule-based baselines")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--interval", default="5m")
    p.add_argument("--period", default="60d")
    p.add_argument("--split", type=float, default=0.8)
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["always_long", "rsi_meanrev"],
        choices=list(POLICIES.keys()),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbols = args.symbols or DEFAULT_SYMBOLS
    data_cfg = DataConfig(interval=args.interval, period=args.period, train_split=args.split)
    env_cfg = EnvConfig()
    dataset = load_dataset(symbols, data_cfg)
    _, eval_data = split_train_eval(dataset, data_cfg.train_split)
    print(f"[baseline] symbols: {list(eval_data.keys())}")

    for strategy in args.strategies:
        policy = POLICIES[strategy]()
        print(f"\n=== baseline: {policy.name} ===")
        summary = run_backtest(policy, eval_data, env_cfg)
        _print_table(summary)


if __name__ == "__main__":
    main()
