"""ScalpingEnv: PPO 단타 환경.

원본: 사용자가 제공한 environment.py.
변경점:
  - 거래 진입 시 페널티 -0.001 -> -0.005 (D2: 잦은 매매 억제 강화)
  - 보유 중 시간 페널티 -0.0001 -> 0 (D3: 보유 자체에 부정적 보상 제거)
  - short 없음 (D1 자동)

Observation:
  최근 N개 봉 OHLCV(가격은 마지막 종가 기준 % 변화) + 5개 기술지표 + 3개 포지션 상태.
Action: 0=Hold, 1=Buy 1주, 2=Sell 전량.
손절/익절은 환경 내부에서 강제 체결.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def calc_rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calc_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    if len(prices) < slow:
        return 0.0, 0.0

    def ema(arr, span):
        alpha = 2 / (span + 1)
        result = arr[0]
        for v in arr[1:]:
            result = alpha * v + (1 - alpha) * result
        return result

    ema_fast = ema(prices[-fast:], fast)
    ema_slow = ema(prices[-slow:], slow)
    macd_line = ema_fast - ema_slow
    return macd_line, macd_line * 0.9


def calc_bollinger(prices: np.ndarray, period: int = 20):
    if len(prices) < period:
        mid = prices[-1]
        return mid, mid + 0.01, mid - 0.01
    window = prices[-period:]
    mid = window.mean()
    std = window.std()
    return mid, mid + 2 * std, mid - 2 * std


def calc_atr(high, low, close, period: int = 14) -> float:
    if len(high) < period + 1:
        return float(high[-1] - low[-1])
    tr_list = []
    for i in range(1, period + 1):
        tr = max(
            high[-i] - low[-i],
            abs(high[-i] - close[-i - 1]),
            abs(low[-i] - close[-i - 1]),
        )
        tr_list.append(tr)
    return float(np.mean(tr_list))


class ScalpingEnv(gym.Env):
    """단타 PPO 환경. df shape (T, 5): open, high, low, close, volume."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: np.ndarray,
        window: int = 30,
        initial_balance: float = 10_000.0,
        commission: float = 0.0025,
        max_position: int = 10,
        stop_loss: float = 0.02,
        take_profit: float = 0.03,
        trade_penalty: float = 0.005,
        hold_penalty: float = 0.0,
        drawdown_threshold: float = 0.05,
        drawdown_penalty: float = 2.0,
    ):
        super().__init__()
        assert df.shape[1] == 5, "df 컬럼 순서: open, high, low, close, volume"
        self.df = df.astype(np.float32)
        self.window = window
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_position = max_position
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trade_penalty = trade_penalty
        self.hold_penalty = hold_penalty
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_penalty = drawdown_penalty

        self.n_features = window * 5 + 5 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.balance = self.initial_balance
        self.position = 0
        self.avg_price = 0.0
        self.total_trades = 0
        self.wins = 0
        self.episode_pnl = 0.0
        self.max_portfolio = self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        start = self.current_step - self.window
        end = self.current_step
        window_data = self.df[start:end]

        opens = window_data[:, 0]
        highs = window_data[:, 1]
        lows = window_data[:, 2]
        closes = window_data[:, 3]
        vols = window_data[:, 4]

        ref = closes[-1] if closes[-1] != 0 else 1.0
        ohlcv_norm = np.concatenate([
            (opens - ref) / ref,
            (highs - ref) / ref,
            (lows - ref) / ref,
            (closes - ref) / ref,
            vols / (vols.mean() + 1e-8) - 1.0,
        ])

        rsi = calc_rsi(closes) / 100.0 - 0.5
        macd_line, macd_sig = calc_macd(closes)
        macd_norm = macd_line / (ref + 1e-8)
        macd_sig_norm = macd_sig / (ref + 1e-8)
        _, bb_up, bb_lo = calc_bollinger(closes)
        bb_width = bb_up - bb_lo
        bb_pct = (closes[-1] - bb_lo) / (bb_width + 1e-8) - 0.5
        atr = calc_atr(highs, lows, closes) / (ref + 1e-8)

        indicators = np.array(
            [rsi, macd_norm, macd_sig_norm, bb_pct, atr], dtype=np.float32
        )

        current_price = closes[-1]
        unrealized_pnl = 0.0
        if self.position > 0 and self.avg_price > 0:
            unrealized_pnl = (current_price - self.avg_price) / self.avg_price

        position_state = np.array(
            [
                self.position / self.max_position,
                unrealized_pnl,
                self.balance / self.initial_balance - 1.0,
            ],
            dtype=np.float32,
        )

        return np.concatenate([ohlcv_norm, indicators, position_state]).astype(np.float32)

    def step(self, action: int):
        current_price = self.df[self.current_step, 3]
        reward = 0.0
        info: dict = {}

        if self.position > 0 and self.avg_price > 0:
            pnl_rate = (current_price - self.avg_price) / self.avg_price
            if pnl_rate <= -self.stop_loss:
                action = 2
                info["stop_loss"] = True
            elif pnl_rate >= self.take_profit:
                action = 2
                info["take_profit"] = True

        if action == 1:
            max_affordable = int(self.balance / (current_price * (1 + self.commission)))
            qty = min(1, max_affordable)
            if qty > 0 and self.position < self.max_position:
                cost = current_price * qty * (1 + self.commission)
                self.balance -= cost
                self.avg_price = (
                    (self.avg_price * self.position + current_price * qty)
                    / (self.position + qty)
                )
                self.position += qty
                self.total_trades += 1
                reward -= self.trade_penalty

        elif action == 2 and self.position > 0:
            qty = self.position
            proceeds = current_price * qty * (1 - self.commission)
            cost_basis = self.avg_price * qty
            trade_pnl = proceeds - cost_basis
            self.balance += proceeds
            self.episode_pnl += trade_pnl
            reward = trade_pnl / self.initial_balance * 100
            if trade_pnl > 0:
                self.wins += 1
            self.position = 0
            self.avg_price = 0.0
            self.total_trades += 1

        else:
            if self.position > 0 and self.hold_penalty > 0:
                reward -= self.hold_penalty

        portfolio_value = self.balance + self.position * current_price
        self.max_portfolio = max(self.max_portfolio, portfolio_value)
        drawdown = (self.max_portfolio - portfolio_value) / self.max_portfolio
        if drawdown > self.drawdown_threshold:
            reward -= drawdown * self.drawdown_penalty

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        if done and self.position > 0:
            final_price = self.df[-1, 3]
            proceeds = final_price * self.position * (1 - self.commission)
            self.balance += proceeds
            self.position = 0

        if done:
            info.update(
                {
                    "episode_pnl": self.episode_pnl,
                    "final_balance": self.balance,
                    "total_trades": self.total_trades,
                    "win_rate": self.wins / max(self.total_trades, 1),
                    "return_pct": (self.balance - self.initial_balance)
                    / self.initial_balance
                    * 100,
                }
            )

        return self._get_obs(), reward, done, False, info

    def render(self, mode: str = "human") -> None:
        current_price = self.df[self.current_step - 1, 3]
        portfolio = self.balance + self.position * current_price
        print(
            f"step={self.current_step} px={current_price:.2f} "
            f"pos={self.position}@{self.avg_price:.2f} "
            f"bal=${self.balance:,.0f} pf=${portfolio:,.0f}"
        )
