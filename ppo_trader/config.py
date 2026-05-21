from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


DEFAULT_SYMBOLS: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    "AMD", "AVGO", "QCOM", "TXN", "INTC", "MU", "AMAT", "LRCX", "ASML",
    "ORCL", "CRM", "ADBE", "NFLX", "PLTR", "SHOP", "UBER", "SNOW", "NOW",
    "JPM", "BAC", "GS", "MS", "C", "V", "MA",
    "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "DIS",
    "JNJ", "PFE", "UNH", "LLY", "ABBV",
    "BA", "CAT", "GE", "XOM", "CVX", "T",
]


@dataclass
class EnvConfig:
    initial_cash: float = 100_000.0
    max_position_pct: float = 1.0
    commission: float = 0.0005
    slippage: float = 0.0003
    window_size: int = 30
    reward_scale: float = 100.0
    drawdown_penalty: float = 0.5
    hold_penalty: float = 0.0001
    turnover_penalty: float = 0.0002
    end_of_day_flat: bool = True
    allow_short: bool = True


@dataclass
class DataConfig:
    interval: str = "5m"
    period: str = "60d"
    train_split: float = 0.8
    cache_dir: str = "data_cache"


@dataclass
class TrainConfig:
    total_timesteps: int = 300_000
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_envs: int = 4
    seed: int = 42
    policy: str = "MlpPolicy"
    policy_kwargs: dict = field(default_factory=lambda: {"net_arch": [256, 256]})
    save_path: str = "models/ppo_daytrader"
    log_path: str = "logs/ppo_daytrader"
