from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ppo_trader.config import EnvConfig
from ppo_trader.features import feature_columns


class DayTradingEnv(gym.Env):
    """Single-asset intraday trading environment.

    Action space (Discrete(3)):
        0 = flat (close any position)
        1 = long
        2 = short (if allow_short)
    Observation:
        Flattened [window_size x n_features] + [position, unrealized_pnl, cash_ratio]
    Reward:
        Δequity (in cash units) * reward_scale - turnover/hold/drawdown penalties.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dataset: Dict[str, pd.DataFrame],
        cfg: Optional[EnvConfig] = None,
        random_symbol: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not dataset:
            raise ValueError("dataset is empty")
        self.dataset = dataset
        self.symbols: List[str] = list(dataset.keys())
        self.cfg = cfg or EnvConfig()
        self.random_symbol = random_symbol

        sample_df = dataset[self.symbols[0]]
        self.feature_cols = feature_columns(sample_df)
        if not self.feature_cols:
            raise ValueError("DataFrame missing attrs['feature_cols']; run add_features().")
        self.n_features = len(self.feature_cols)

        self.action_space = spaces.Discrete(3 if self.cfg.allow_short else 2)
        obs_dim = self.cfg.window_size * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = random.Random(seed)
        self._reset_state()

    def _reset_state(self) -> None:
        self.current_symbol: str = ""
        self.df: pd.DataFrame = pd.DataFrame()
        self.feature_matrix: np.ndarray = np.empty((0, self.n_features), dtype=np.float32)
        self.timestamps: pd.DatetimeIndex = pd.DatetimeIndex([])
        self.t: int = 0
        self.position: int = 0
        self.entry_price: float = 0.0
        self.cash: float = self.cfg.initial_cash
        self.equity: float = self.cfg.initial_cash
        self.peak_equity: float = self.cfg.initial_cash
        self.shares: float = 0.0
        self.history: List[Dict[str, float]] = []

    def _pick_symbol(self) -> str:
        if self.random_symbol and len(self.symbols) > 1:
            return self._rng.choice(self.symbols)
        return self.symbols[0]

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng.seed(seed)
        super().reset(seed=seed)
        self._reset_state()
        self.current_symbol = self._pick_symbol()
        self.df = self.dataset[self.current_symbol]
        self.feature_matrix = self.df[self.feature_cols].to_numpy(dtype=np.float32)
        self.timestamps = self.df.index
        self.t = self.cfg.window_size
        return self._obs(), {"symbol": self.current_symbol}

    def _price(self, idx: Optional[int] = None) -> float:
        i = self.t if idx is None else idx
        return float(self.df["close"].iat[i])

    def _is_session_end(self, idx: int) -> bool:
        if idx >= len(self.timestamps) - 1:
            return True
        if not self.cfg.end_of_day_flat:
            return False
        return self.timestamps[idx].date() != self.timestamps[idx + 1].date()

    def _mark_to_market(self) -> float:
        price = self._price()
        position_value = self.shares * price
        return self.cash + position_value

    def _apply_action(self, action: int) -> Tuple[float, float]:
        """Execute target position change. Returns (turnover_notional, realized_pnl)."""
        price = self._price()
        target = 0
        if action == 1:
            target = 1
        elif action == 2 and self.cfg.allow_short:
            target = -1

        if target == self.position:
            return 0.0, 0.0

        realized = 0.0
        turnover = 0.0

        if self.position != 0:
            close_price = price * (1 - self.cfg.slippage * np.sign(self.shares))
            gross = self.shares * close_price
            fee = abs(gross) * self.cfg.commission
            self.cash += gross - fee
            realized = (close_price - self.entry_price) * self.shares
            turnover += abs(gross)
            self.shares = 0.0
            self.position = 0
            self.entry_price = 0.0

        if target != 0:
            notional = self.cash * self.cfg.max_position_pct
            exec_price = price * (1 + self.cfg.slippage * target)
            qty = (notional / exec_price) * target
            cost = qty * exec_price
            fee = abs(cost) * self.cfg.commission
            self.cash -= cost + fee
            self.shares = qty
            self.position = target
            self.entry_price = exec_price
            turnover += abs(cost)

        return turnover, realized

    def _obs(self) -> np.ndarray:
        start = self.t - self.cfg.window_size
        window = self.feature_matrix[start : self.t]
        flat = window.reshape(-1)
        price = self._price()
        unrealized = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized = (price / self.entry_price - 1.0) * self.position
        cash_ratio = self.cash / max(self.equity, 1e-6)
        extras = np.array([self.position, unrealized, cash_ratio - 1.0], dtype=np.float32)
        return np.concatenate([flat, extras]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.t >= len(self.df) - 1:
            return self._obs(), 0.0, True, False, {"reason": "out_of_data"}

        prev_equity = self._mark_to_market()
        turnover, realized = self._apply_action(int(action))

        self.t += 1
        new_equity = self._mark_to_market()

        delta = new_equity - prev_equity
        self.equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)
        drawdown = max(0.0, (self.peak_equity - new_equity) / max(self.peak_equity, 1e-6))

        reward = delta / max(self.cfg.initial_cash, 1e-6) * self.cfg.reward_scale
        reward -= self.cfg.turnover_penalty * (turnover / max(self.cfg.initial_cash, 1e-6))
        reward -= self.cfg.drawdown_penalty * drawdown
        if self.position == 0:
            reward -= self.cfg.hold_penalty

        terminated = False
        truncated = False
        info: Dict[str, float] = {
            "equity": new_equity,
            "delta": delta,
            "position": self.position,
            "drawdown": drawdown,
            "realized": realized,
            "turnover": turnover,
            "price": self._price(),
            "symbol": self.current_symbol,
        }

        if self._is_session_end(self.t) and self.position != 0:
            _, realized_eod = self._apply_action(0)
            info["realized"] = realized + realized_eod
            new_equity = self._mark_to_market()
            self.equity = new_equity

        if self.t >= len(self.df) - 1:
            terminated = True

        self.history.append(
            {
                "t": int(self.t),
                "equity": float(new_equity),
                "position": int(self.position),
                "price": float(self._price()),
            }
        )
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        print(
            f"[{self.current_symbol}] t={self.t} px={self._price():.2f} "
            f"pos={self.position} equity={self.equity:,.2f}"
        )

    def config_dict(self) -> dict:
        return asdict(self.cfg)
