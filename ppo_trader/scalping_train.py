"""ScalpingEnv 기반 PPO 학습 스크립트.

원본: 사용자가 제공한 model.py.
변경점:
  - 데이터 소스 alpaca / yfinance 양쪽 지원 (DATA_SOURCE 환경변수)
  - 다중 종목 학습 지원 (--symbols A B C ...) — VecEnv에 종목별 환경 1개씩
  - 기존 모델 있으면 이어 학습 (resume)
  - 모델 저장 경로 단일화 (--save-path)
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import List, Sequence

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from ppo_trader.scalping_env import ScalpingEnv


# ── 데이터 로딩 ────────────────────────────────────────────────


def _ohlcv_to_array(df: pd.DataFrame) -> np.ndarray:
    cols = ["open", "high", "low", "close", "volume"]
    df = df[cols].dropna()
    return df.to_numpy(dtype=np.float32)


def fetch_ohlcv(
    symbol: str,
    interval: str,
    period: str,
    cache_dir: str = "data_cache",
) -> np.ndarray:
    """Use ppo_trader.data.fetch_symbol then return raw OHLCV ndarray."""
    from ppo_trader.data import fetch_symbol

    df = fetch_symbol(
        symbol=symbol,
        interval=interval,
        period=period,
        cache_dir=cache_dir,
        use_cache=True,
    )
    arr = _ohlcv_to_array(df)
    if len(arr) < 200:
        raise RuntimeError(f"{symbol}: {len(arr)} bars (need >= 200)")
    return arr


def train_test_split(data: np.ndarray, test_ratio: float = 0.2):
    split = int(len(data) * (1 - test_ratio))
    return data[:split], data[split:]


# ── 콜백 ───────────────────────────────────────────────────────


class TrainingCallback(BaseCallback):
    def __init__(self, print_every: int = 20_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = print_every
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0])
        self.current_episode_reward += float(rewards[0])
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
        if self.n_calls % self.print_every == 0 and self.episode_rewards:
            recent = self.episode_rewards[-20:]
            print(
                f"  Step {self.n_calls:>8,} | "
                f"recent ep mean reward {np.mean(recent):+.4f} | "
                f"episodes {len(self.episode_rewards)}"
            )
        return True


# ── 환경 팩토리 ────────────────────────────────────────────────


def make_env_fn(data: np.ndarray, **kwargs):
    def _init():
        env = ScalpingEnv(data, **kwargs)
        env = Monitor(env)
        return env

    return _init


def build_vec_env(
    train_arrays: Sequence[np.ndarray],
    n_envs: int,
    env_kwargs: dict,
):
    factories = []
    for i in range(n_envs):
        data = train_arrays[i % len(train_arrays)]
        factories.append(make_env_fn(data, **env_kwargs))
    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    return VecMonitor(vec_cls(factories))


# ── 백테스트 ───────────────────────────────────────────────────


def backtest(model: PPO, test_data: np.ndarray, env_kwargs: dict, label: str) -> dict:
    env = ScalpingEnv(test_data, **env_kwargs)
    obs, _ = env.reset()
    portfolio_history: List[float] = []
    action_counts = {0: 0, 1: 0, 2: 0}

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(action))
        action_counts[int(action)] += 1
        current_price = test_data[env.current_step - 1, 3]
        portfolio = env.balance + env.position * current_price
        portfolio_history.append(portfolio)

    initial = env_kwargs["initial_balance"]
    final = env.balance
    ret = (final - initial) / initial * 100
    pf = np.array(portfolio_history)
    rolling_max = np.maximum.accumulate(pf)
    drawdowns = (rolling_max - pf) / rolling_max
    max_dd = float(drawdowns.max() * 100)
    returns = np.diff(pf) / pf[:-1] if len(pf) > 1 else np.array([0.0])
    sharpe = float((returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252 * 78))
    wins = env.wins
    total_trades = env.total_trades
    win_rate = wins / max(total_trades, 1) * 100

    print(f"\n[{label}] backtest")
    print(f"  return     {ret:+.2f}%")
    print(f"  max DD     {max_dd:.2f}%")
    print(f"  sharpe     {sharpe:.2f}")
    print(f"  trades     {total_trades}  (win {win_rate:.1f}%)")
    print(
        f"  actions    hold={action_counts[0]} buy={action_counts[1]} sell={action_counts[2]}"
    )

    return {
        "symbol": label,
        "return_pct": ret,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "trades": total_trades,
        "win_rate": win_rate,
        "final_balance": final,
    }


# ── 메인 ───────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ScalpingEnv PPO trainer")
    p.add_argument("--symbols", nargs="+", default=["AAPL"])
    p.add_argument("--interval", default="5m")
    p.add_argument("--period", default="60d")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--balance", type=float, default=10_000.0)
    p.add_argument("--commission", type=float, default=0.0025)
    p.add_argument("--max-position", type=int, default=10)
    p.add_argument("--stop-loss", type=float, default=0.02)
    p.add_argument("--take-profit", type=float, default=0.03)
    p.add_argument("--trade-penalty", type=float, default=0.005)
    p.add_argument("--hold-penalty", type=float, default=0.0)
    p.add_argument("--save-path", default="models/ppo_scalper")
    p.add_argument("--log-dir", default="logs/scalper")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"[scalping] symbols={args.symbols} interval={args.interval} "
        f"period={args.period} timesteps={args.timesteps:,} n_envs={args.n_envs}"
    )

    train_arrays: List[np.ndarray] = []
    test_arrays: List[tuple[str, np.ndarray]] = []
    for sym in args.symbols:
        try:
            arr = fetch_ohlcv(sym, args.interval, args.period)
        except Exception as exc:
            print(f"[scalping] skip {sym}: {exc}")
            continue
        tr, te = train_test_split(arr, test_ratio=0.2)
        print(f"[scalping] {sym}: {len(tr):,} train / {len(te):,} test bars")
        train_arrays.append(tr)
        test_arrays.append((sym, te))

    if not train_arrays:
        raise RuntimeError("No usable symbols")

    env_kwargs = dict(
        window=args.window,
        initial_balance=args.balance,
        commission=args.commission,
        max_position=args.max_position,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        trade_penalty=args.trade_penalty,
        hold_penalty=args.hold_penalty,
    )

    vec_env = build_vec_env(train_arrays, args.n_envs, env_kwargs)
    eval_env = VecMonitor(DummyVecEnv([make_env_fn(test_arrays[0][1], **env_kwargs)]))

    existing = Path(args.save_path + ".zip")
    reset_timesteps = True
    if existing.exists():
        print(f"[scalping] resuming from {existing}")
        model = PPO.load(
            str(existing),
            env=vec_env,
            tensorboard_log=args.log_dir,
            verbose=0,
        )
        reset_timesteps = False
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256, 128]),
            verbose=0,
            tensorboard_log=args.log_dir,
        )

    callbacks = [
        TrainingCallback(print_every=20_000),
        CheckpointCallback(
            save_freq=max(50_000 // max(args.n_envs, 1), 1),
            save_path=args.log_dir,
            name_prefix="ppo_ckpt",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=args.log_dir,
            log_path=args.log_dir,
            eval_freq=max(20_000 // max(args.n_envs, 1), 1),
            n_eval_episodes=2,
            deterministic=True,
            verbose=0,
        ),
    ]

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_timesteps,
        progress_bar=False,
    )

    model.save(args.save_path)
    print(f"[scalping] saved model to {args.save_path}.zip (resume={not reset_timesteps})")

    print("\n=== held-out backtest ===")
    rows: List[dict] = []
    for sym, te in test_arrays:
        try:
            rows.append(backtest(model, te, env_kwargs, sym))
        except Exception as exc:
            print(f"[scalping] backtest failed for {sym}: {exc}")

    if rows:
        df = pd.DataFrame(rows)
        avg = df[["return_pct", "max_drawdown_pct", "sharpe", "win_rate"]].mean()
        print("\n=== average across symbols ===")
        print(
            f"  return     {avg['return_pct']:+.2f}%\n"
            f"  max DD     {avg['max_drawdown_pct']:.2f}%\n"
            f"  sharpe     {avg['sharpe']:.2f}\n"
            f"  win rate   {avg['win_rate']:.1f}%"
        )


if __name__ == "__main__":
    main()
