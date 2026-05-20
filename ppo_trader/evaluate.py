from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ppo_trader.config import DEFAULT_SYMBOLS, DataConfig, EnvConfig
from ppo_trader.data import load_dataset, split_train_eval
from ppo_trader.env import DayTradingEnv


def _metrics(equity_curve: List[float], initial: float) -> Dict[str, float]:
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) < 2:
        return {"return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe": 0.0, "n": len(eq)}
    rets = np.diff(eq) / eq[:-1]
    sharpe = 0.0
    if rets.std(ddof=0) > 0:
        sharpe = float(rets.mean() / rets.std(ddof=0) * np.sqrt(252 * 78))  # ~5m bars/day
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    return {
        "return_pct": float((eq[-1] / initial - 1.0) * 100.0),
        "max_drawdown_pct": float(dd.max() * 100.0),
        "sharpe": sharpe,
        "n": int(len(eq)),
        "final_equity": float(eq[-1]),
    }


def run_backtest(
    model_path: str,
    dataset: Dict[str, pd.DataFrame],
    env_cfg: EnvConfig,
    deterministic: bool = True,
) -> Dict[str, Dict[str, float]]:
    model = PPO.load(model_path)
    summary: Dict[str, Dict[str, float]] = {}

    for symbol, df in dataset.items():
        env = DayTradingEnv(
            dataset={symbol: df}, cfg=env_cfg, random_symbol=False, seed=0
        )
        obs, _ = env.reset()
        equity_curve: List[float] = [env.equity]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(int(action))
            equity_curve.append(env.equity)
            done = terminated or truncated
        summary[symbol] = _metrics(equity_curve, env_cfg.initial_cash)
        summary[symbol]["bars"] = len(df)

    return summary


def _print_table(summary: Dict[str, Dict[str, float]]) -> None:
    cols = ["symbol", "return_pct", "max_drawdown_pct", "sharpe", "final_equity", "bars"]
    print(" | ".join(f"{c:>16}" for c in cols))
    print("-" * (19 * len(cols)))
    totals = {"return_pct": [], "max_drawdown_pct": [], "sharpe": []}
    for s, m in summary.items():
        row = [s, f"{m['return_pct']:.2f}", f"{m['max_drawdown_pct']:.2f}",
               f"{m['sharpe']:.2f}", f"{m['final_equity']:,.0f}", str(int(m.get('bars', 0)))]
        print(" | ".join(f"{v:>16}" for v in row))
        for k in totals:
            totals[k].append(m[k])
    print("-" * (19 * len(cols)))
    if totals["return_pct"]:
        avg = {k: float(np.mean(v)) for k, v in totals.items()}
        print(
            f"{'AVERAGE':>16} | {avg['return_pct']:>16.2f} | "
            f"{avg['max_drawdown_pct']:>16.2f} | {avg['sharpe']:>16.2f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO day trader on held-out data")
    p.add_argument("--model", default="models/ppo_daytrader")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--interval", default="5m")
    p.add_argument("--period", default="60d")
    p.add_argument("--split", type=float, default=0.8)
    p.add_argument("--stochastic", action="store_true", help="sample instead of argmax")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbols = args.symbols or DEFAULT_SYMBOLS
    data_cfg = DataConfig(interval=args.interval, period=args.period, train_split=args.split)
    env_cfg = EnvConfig()
    dataset = load_dataset(symbols, data_cfg)
    _, eval_data = split_train_eval(dataset, data_cfg.train_split)
    print(f"[eval] symbols: {list(eval_data.keys())}")
    summary = run_backtest(args.model, eval_data, env_cfg, deterministic=not args.stochastic)
    _print_table(summary)


if __name__ == "__main__":
    main()
