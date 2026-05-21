from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from ppo_trader.config import DEFAULT_SYMBOLS, DataConfig, EnvConfig, TrainConfig
from ppo_trader.data import load_dataset, split_train_eval
from ppo_trader.env import DayTradingEnv


def _make_env_factory(dataset: Dict[str, pd.DataFrame], env_cfg: EnvConfig, seed: int):
    def _thunk():
        env = DayTradingEnv(dataset=dataset, cfg=env_cfg, random_symbol=True, seed=seed)
        return Monitor(env)
    return _thunk


def train(
    symbols=None,
    data_cfg: DataConfig = DataConfig(),
    env_cfg: EnvConfig = EnvConfig(),
    train_cfg: TrainConfig = TrainConfig(),
) -> str:
    symbols = symbols or DEFAULT_SYMBOLS
    print(f"[train] loading {len(symbols)} symbols ({data_cfg.interval}, {data_cfg.period})")
    dataset = load_dataset(symbols, data_cfg)
    train_data, eval_data = split_train_eval(dataset, data_cfg.train_split)
    print(f"[train] usable symbols: {list(train_data.keys())}")

    Path(train_cfg.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(train_cfg.log_path).mkdir(parents=True, exist_ok=True)

    env_fns = [
        _make_env_factory(train_data, env_cfg, train_cfg.seed + i)
        for i in range(train_cfg.n_envs)
    ]
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    vec_cls = DummyVecEnv if train_cfg.n_envs == 1 else SubprocVecEnv
    vec_env = VecMonitor(vec_cls(env_fns))

    eval_env = VecMonitor(DummyVecEnv([_make_env_factory(eval_data, env_cfg, train_cfg.seed + 999)]))

    existing = Path(train_cfg.save_path + ".zip")
    reset_timesteps = True
    if existing.exists():
        print(f"[train] resuming from {existing}")
        model = PPO.load(
            str(existing),
            env=vec_env,
            tensorboard_log=train_cfg.log_path,
            verbose=1,
        )
        reset_timesteps = False
    else:
        model = PPO(
            train_cfg.policy,
            vec_env,
            learning_rate=train_cfg.learning_rate,
            n_steps=train_cfg.n_steps,
            batch_size=train_cfg.batch_size,
            n_epochs=train_cfg.n_epochs,
            gamma=train_cfg.gamma,
            gae_lambda=train_cfg.gae_lambda,
            clip_range=train_cfg.clip_range,
            ent_coef=train_cfg.ent_coef,
            vf_coef=train_cfg.vf_coef,
            max_grad_norm=train_cfg.max_grad_norm,
            policy_kwargs=train_cfg.policy_kwargs,
            tensorboard_log=train_cfg.log_path,
            seed=train_cfg.seed,
            verbose=1,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(10_000 // max(train_cfg.n_envs, 1), 1),
            save_path=train_cfg.log_path,
            name_prefix="ppo_ckpt",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=train_cfg.log_path,
            log_path=train_cfg.log_path,
            eval_freq=max(20_000 // max(train_cfg.n_envs, 1), 1),
            deterministic=True,
            render=False,
        ),
    ]

    model.learn(
        total_timesteps=train_cfg.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_timesteps,
    )
    model.save(train_cfg.save_path)
    print(f"[train] saved model to {train_cfg.save_path}.zip (resume={not reset_timesteps})")
    return train_cfg.save_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a PPO US-stock day trader")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--interval", default="5m")
    p.add_argument("--period", default="60d")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--allow-short", action="store_true", default=True)
    p.add_argument("--no-short", dest="allow_short", action="store_false")
    p.add_argument("--save-path", default="models/ppo_daytrader")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = DataConfig(interval=args.interval, period=args.period)
    env_cfg = EnvConfig(allow_short=args.allow_short)
    train_cfg = TrainConfig(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_path=args.save_path,
    )
    train(args.symbols, data_cfg, env_cfg, train_cfg)


if __name__ == "__main__":
    main()
