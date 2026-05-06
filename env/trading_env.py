"""
NSE intraday trading simulation environment for SAFE-HFT.

Implements a Gymnasium-compatible environment that replays minute-level OHLCV data
for a single NSE equity. One episode = one trading day. The agent receives a
flattened observation vector, issues discrete actions, and receives a reward equal
to the mark-to-market log-return of its position minus transaction costs and slippage.
Risk is returned as a separate cost dict in ``info`` — it is NEVER added to reward.

Observation layout (flat float32, total 290 values)
-----------------------------------------------------
  [0   : 280]  Last 20 bars x 14 numeric features (flattened row-major)
  [280 : 283]  Current position one-hot  [short, flat, long]
  [283]        Normalised unrealised PnL  (pnl / initial_capital)
  [284]        Current drawdown from episode peak  (in [0, 1])
  [285 : 287]  Time-of-day encoding  [sin(2pi*t/375), cos(2pi*t/375)]
  [287 : 290]  Market regime one-hot  [trending_up, trending_down, high_vol]

Action space  Discrete(4)
  0 — Hold
  1 — Buy   (go long;  close short first if needed)
  2 — Sell  (go short; close long  first if needed)
  3 — Close (flatten current position)

Reward
  r_t = position_direction * log(close[t] / close[t-1])
        - transaction_cost * is_trade_t
        - slippage * is_trade_t
  Risk is NOT penalised in reward. Risk signals live in info["costs"].

Author: [Author Name]
"""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from env.market_rules import NSEMarketRules, TOTAL_SESSION_MINUTES, INTRADAY_CLOSE_MINUTE
from constraints.risk_constraints import (
    compute_drawdown_cost,
    compute_duration_cost,
    compute_volatility_cost,
)

# Feature columns used in the 20-bar lookback (excludes 'regime', which is one-hot separately)
LOOKBACK_FEATURE_COLS: list[str] = [
    "log_return", "volatility_10", "volatility_30", "volatility_60",
    "rsi_14", "macd", "macd_signal", "macd_hist", "bb_width_20",
    "volume_ratio_30", "spread_proxy", "price_impact",
    "volume_imbalance", "ob_imbalance",
]
N_LOOKBACK_FEATURES: int = len(LOOKBACK_FEATURE_COLS)   # 14
LOOKBACK_WINDOW: int = 20
OBS_SIZE: int = LOOKBACK_WINDOW * N_LOOKBACK_FEATURES + 3 + 1 + 1 + 2 + 3  # 290

# Position constants
POS_SHORT: int = -1
POS_FLAT: int = 0
POS_LONG: int = 1

# Action constants
ACT_HOLD: int = 0
ACT_BUY: int = 1
ACT_SELL: int = 2
ACT_CLOSE: int = 3


class NSETradingEnv(gym.Env):
    """
    Gymnasium environment for single-stock NSE intraday trading.

    One episode equals one trading day. The environment cycles through days
    randomly during training and sequentially during validation/testing.

    Parameters
    ----------
    data : pd.DataFrame
        Processed feature DataFrame for one ticker. Must contain OHLCV columns
        plus all columns in ``LOOKBACK_FEATURE_COLS`` and ``regime``.
    split_info : dict
        Split boundary dict as produced by ``DataPreprocessor.save_split_indices``.
        Keys: ``"train"``, ``"val"``, ``"test"``, each with ``"start"`` / ``"end"``.
    config : dict
        Env section from ``config.yaml``.
    ticker : str
        Ticker symbol (used for logging).
    mode : str
        One of ``"train"``, ``"val"``, ``"test"``.
    seed : int
        Master random seed.
    """

    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        split_info: dict,
        config: dict,
        ticker: str = "RELIANCE",
        mode: str = "train",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.ticker = ticker
        self.mode = mode
        self._seed = seed
        self.config = config

        # Slice data to the requested split
        start = pd.Timestamp(split_info[mode]["start"])
        end = pd.Timestamp(split_info[mode]["end"])
        self.data = data.loc[start:end].copy()

        if self.data.empty:
            raise ValueError(
                f"No data for mode='{mode}' between {start.date()} and {end.date()}"
            )

        # Build list of trading days in this split
        self._dates: list[pd.Timestamp] = sorted(
            self.data.index.normalize().unique().tolist()
        )
        self._n_days: int = len(self._dates)
        self._day_cursor: int = 0   # used in val/test for sequential iteration

        # Episode state (populated in reset())
        self._episode_bars: pd.DataFrame = pd.DataFrame()
        # Fast numpy views of episode data — populated in reset()
        self._ep_close: np.ndarray = np.empty(0, dtype=np.float64)
        self._ep_log_return: np.ndarray = np.empty(0, dtype=np.float64)
        self._ep_spread_proxy: np.ndarray = np.empty(0, dtype=np.float64)
        self._ep_regime: np.ndarray = np.empty(0, dtype=np.int32)
        self._ep_features: np.ndarray = np.empty((0, N_LOOKBACK_FEATURES), dtype=np.float32)
        self._current_step: int = 0
        self._position: int = POS_FLAT
        self._entry_price: float = 0.0
        self._portfolio_value: float = 1.0
        self._peak_portfolio_value: float = 1.0
        self._position_duration: int = 0
        self._cumulative_daily_return: float = 0.0
        self._agent_returns: list[float] = []
        self._lookback_buffer: np.ndarray = np.zeros(
            (LOOKBACK_WINDOW, N_LOOKBACK_FEATURES), dtype=np.float32
        )
        self._current_regime: int = 2

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # NSE rules helper (stateless)
        self._rules = NSEMarketRules()

        # RNG
        self._rng = np.random.default_rng(seed)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Start a new episode (one trading day).

        Parameters
        ----------
        seed : int, optional
        options : dict, optional

        Returns
        -------
        obs : np.ndarray
        info : dict
        """
        super().reset(seed=seed)   # registers self.np_random for gymnasium compliance
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        # Pick the day for this episode
        if self.mode == "train":
            day_idx = int(self._rng.integers(0, self._n_days))
        else:
            day_idx = self._day_cursor % self._n_days
            self._day_cursor += 1

        target_date = self._dates[day_idx]
        day_mask = self.data.index.normalize() == target_date
        self._episode_bars = self.data.loc[day_mask].copy()

        if len(self._episode_bars) < 2:
            # Degenerate day — try the next one
            self._day_cursor += 1
            return self.reset(seed=seed, options=options)

        # Pre-convert episode columns to contiguous numpy arrays for fast step() access
        self._ep_close = self._episode_bars["close"].to_numpy(dtype=np.float64, na_value=0.0)
        self._ep_log_return = self._episode_bars["log_return"].to_numpy(dtype=np.float64, na_value=0.0)
        self._ep_spread_proxy = self._episode_bars["spread_proxy"].to_numpy(dtype=np.float64, na_value=0.0)
        regime_raw = self._episode_bars["regime"].fillna(2).to_numpy(dtype=np.float64)
        self._ep_regime = regime_raw.astype(np.int32)
        self._ep_features = np.nan_to_num(
            self._episode_bars[LOOKBACK_FEATURE_COLS].to_numpy(dtype=np.float32),
            nan=0.0,
        )

        # Seed the lookback buffer from the 20 bars preceding this day in the dataset
        self._init_lookback_buffer(target_date)

        # Reset episode state
        self._current_step = 0
        self._position = POS_FLAT
        self._entry_price = 0.0
        self._portfolio_value = 1.0
        self._peak_portfolio_value = 1.0
        self._position_duration = 0
        self._cumulative_daily_return = 0.0
        self._agent_returns = []
        self._current_regime = int(
            self._episode_bars.iloc[0].get("regime", 2) or 2
        )

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance the environment by one bar.

        Parameters
        ----------
        action : int
            Raw agent action in ``{0, 1, 2, 3}``.

        Returns
        -------
        obs : np.ndarray
        reward : float
        terminated : bool
        truncated : bool
        info : dict
            Includes ``costs`` dict with binary constraint signals.
        """
        s = self._current_step
        minute_index = s  # 0 = 9:15 AM bar

        # Apply NSE market rules (may override action)
        action = self._rules.apply_circuit_breaker(
            action, self._cumulative_daily_return
        )
        action = self._rules.apply_intraday_close(
            action, minute_index, self._position
        )

        # Price for this bar — use pre-converted numpy array
        price_now = self._ep_close[s]
        price_prev = self._ep_close[s - 1] if s > 0 else price_now

        # Execute action and compute reward
        reward, trade_occurred = self._execute_action(
            action, price_now, price_prev, s
        )

        # Update position duration counter
        if self._position != POS_FLAT:
            self._position_duration += 1
        else:
            self._position_duration = 0

        # Update portfolio value and drawdown
        self._portfolio_value *= (1.0 + reward)
        self._peak_portfolio_value = max(
            self._peak_portfolio_value, self._portfolio_value
        )

        # Track agent's per-bar return for volatility constraint
        self._agent_returns.append(reward)

        # Update market return for volatility constraint
        market_log_return = self._ep_log_return[s]
        self._cumulative_daily_return += market_log_return

        # Compute safety constraint cost signals
        costs = self._compute_cost_signals(s)

        # Advance lookback buffer
        self._update_lookback_buffer(s)
        self._current_step += 1

        # Update regime for observation
        self._current_regime = int(self._ep_regime[s])

        terminated = self._current_step >= len(self._episode_bars)
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(OBS_SIZE, dtype=np.float32)

        info = {
            "costs": costs,
            "total_cost": float(sum(costs.values())),
            "position": self._position,
            "portfolio_value": self._portfolio_value,
            "drawdown": self._current_drawdown(),
            "trade": trade_occurred,
            "minute_index": minute_index,
            "regime": self._current_regime,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Print a one-line status summary."""
        step = self._current_step
        pos_str = {POS_LONG: "LONG", POS_FLAT: "FLAT", POS_SHORT: "SHORT"}
        print(
            f"[{self.ticker}] step={step:3d}  "
            f"pos={pos_str.get(self._position,'?'):5s}  "
            f"pv={self._portfolio_value:.4f}  "
            f"dd={self._current_drawdown():.4f}  "
            f"regime={self._current_regime}"
        )

    def close(self) -> None:
        pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _execute_action(
        self,
        action: int,
        price_now: float,
        price_prev: float,
        step_idx: int,
    ) -> tuple[float, bool]:
        """
        Update position state and compute the per-step reward.

        Returns
        -------
        reward : float
        trade_occurred : bool
        """
        trade_occurred = False
        trade_cost = 0.0
        spread_proxy = self._ep_spread_proxy[step_idx]
        # De-normalise spread proxy is tricky post z-score; use a floor instead
        raw_spread = max(spread_proxy * 0.0001, 0.0)

        prev_position = self._position

        # Determine new position
        if action == ACT_BUY:
            new_position = POS_LONG
        elif action == ACT_SELL:
            new_position = POS_SHORT
        elif action == ACT_CLOSE:
            new_position = POS_FLAT
        else:  # HOLD
            new_position = self._position

        # Trade costs apply when position changes
        if new_position != prev_position:
            trade_occurred = True
            tc = self._rules.compute_transaction_cost(price_now)
            sl = self._rules.compute_slippage(raw_spread)
            trade_cost = tc + sl
            self._position = new_position
            if new_position != POS_FLAT:
                self._entry_price = price_now
            else:
                self._entry_price = 0.0
            self._position_duration = 0

        # Mark-to-market reward for the NEW position (after the trade)
        log_ret = np.log(price_now / price_prev) if price_prev > 0 else 0.0
        reward = self._position * log_ret - trade_cost

        return reward, trade_occurred

    def _compute_cost_signals(self, step_idx: int) -> dict[str, float]:
        """
        Evaluate all three safety constraints for the current timestep.

        Returns
        -------
        dict with keys ``cost_drawdown``, ``cost_duration``, ``cost_volatility``.
        """
        cfg = self.config
        n = len(self._agent_returns)
        agent_returns = np.array(self._agent_returns, dtype=np.float64)
        # Market returns slice — use pre-computed numpy array (no pandas)
        vol_window = int(cfg.get("vol_window", 60))
        start = max(0, n - vol_window)
        market_returns = self._ep_log_return[start:n]

        c_dd = compute_drawdown_cost(
            self._peak_portfolio_value,
            self._portfolio_value,
            threshold=float(cfg.get("max_drawdown_threshold", 0.05)),
        )
        c_dur = compute_duration_cost(
            self._position_duration,
            duration_limit=int(cfg.get("duration_limit", 30)),
        )
        c_vol = compute_volatility_cost(
            agent_returns[-vol_window:] if len(agent_returns) >= vol_window else agent_returns,
            market_returns,
            window=vol_window,
            vol_multiplier=float(cfg.get("vol_multiplier", 2.0)),
        )
        return {
            "cost_drawdown": c_dd,
            "cost_duration": c_dur,
            "cost_volatility": c_vol,
        }

    def _get_obs(self) -> np.ndarray:
        """
        Build the current 290-element observation vector.

        Returns
        -------
        np.ndarray  shape (290,), dtype float32
        """
        obs = np.empty(OBS_SIZE, dtype=np.float32)

        # 1. 20 × 14 lookback features (already updated to include current bar)
        obs[:LOOKBACK_WINDOW * N_LOOKBACK_FEATURES] = (
            self._lookback_buffer.flatten()
        )

        base = LOOKBACK_WINDOW * N_LOOKBACK_FEATURES  # 280

        # 2. Position one-hot [short, flat, long]
        obs[base] = float(self._position == POS_SHORT)
        obs[base + 1] = float(self._position == POS_FLAT)
        obs[base + 2] = float(self._position == POS_LONG)

        # 3. Normalised unrealised PnL
        obs[base + 3] = float(self._portfolio_value - 1.0)

        # 4. Current drawdown from peak
        obs[base + 4] = float(self._current_drawdown())

        # 5. Time-of-day encoding
        t = self._current_step
        obs[base + 5] = np.sin(2 * np.pi * t / TOTAL_SESSION_MINUTES)
        obs[base + 6] = np.cos(2 * np.pi * t / TOTAL_SESSION_MINUTES)

        # 6. Regime one-hot [trending_up, trending_down, high_vol]
        obs[base + 7] = float(self._current_regime == 0)
        obs[base + 8] = float(self._current_regime == 1)
        obs[base + 9] = float(self._current_regime == 2)

        return obs

    def _init_lookback_buffer(self, target_date: pd.Timestamp) -> None:
        """
        Seed the 20-bar lookback buffer from data BEFORE *target_date*.

        Uses the last 20 available bars in the dataset that precede this day.
        Pads with zeros if fewer than 20 prior bars exist.
        """
        prior = self.data.loc[self.data.index < target_date]
        if len(prior) >= LOOKBACK_WINDOW:
            rows = prior.iloc[-LOOKBACK_WINDOW:]
            self._lookback_buffer = (
                rows[LOOKBACK_FEATURE_COLS].values.astype(np.float32)
            )
        else:
            n = len(prior)
            self._lookback_buffer = np.zeros(
                (LOOKBACK_WINDOW, N_LOOKBACK_FEATURES), dtype=np.float32
            )
            if n > 0:
                rows = prior.iloc[-n:]
                self._lookback_buffer[-n:] = (
                    rows[LOOKBACK_FEATURE_COLS].values.astype(np.float32)
                )
        # Replace any NaN that survived
        np.nan_to_num(self._lookback_buffer, nan=0.0, copy=False)

    def _update_lookback_buffer(self, step_idx: int) -> None:
        """Shift the lookback buffer left by one and append the current bar's features."""
        self._lookback_buffer[:-1] = self._lookback_buffer[1:]
        self._lookback_buffer[-1] = self._ep_features[step_idx]

    def _current_drawdown(self) -> float:
        """Return current drawdown from peak as a fraction in [0, 1]."""
        if self._peak_portfolio_value <= 0:
            return 0.0
        dd = (self._peak_portfolio_value - self._portfolio_value) / self._peak_portfolio_value
        return float(np.clip(dd, 0.0, 1.0))

    def _time_encoding(self, minute_index: int) -> np.ndarray:
        """
        Sinusoidal time-of-day encoding.

        Parameters
        ----------
        minute_index : int

        Returns
        -------
        np.ndarray  shape (2,)
        """
        angle = 2 * np.pi * minute_index / TOTAL_SESSION_MINUTES
        return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
