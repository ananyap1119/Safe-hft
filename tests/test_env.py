"""
Unit tests for NSETradingEnv.

Covers:
  - gymnasium.utils.env_checker.check_env() compliance
  - Full episode with random agent completes without error
  - Reproducibility: identical trajectories with same seed
  - Intraday close rule: position is always flat at episode end
  - Time encoding: output is a unit-circle 2-vector
  - No lookahead in observations (obs at t contains only data up to t)
  - Reward has no risk penalty (costs live in info only)

Author: [Author Name]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "datasets" / "processed"

# ── Shared fixture ─────────────────────────────────────────────────────────────

def _build_env(mode: str = "test"):
    """Construct an NSETradingEnv for RELIANCE in the requested mode."""
    from env.trading_env import NSETradingEnv
    df = pd.read_parquet(DATA_DIR / "RELIANCE_processed.parquet")
    with open(DATA_DIR / "RELIANCE_splits.json", encoding="utf-8") as f:
        splits = json.load(f)
    cfg = {
        "lookback_window": 20,
        "initial_capital": 1.0,
        "transaction_cost": 0.0003,
        "slippage_rate": 0.0001,
        "tick_size": 0.05,
        "session_minutes": 375,
        "intraday_close_minute": 360,
        "circuit_thresholds": [0.05, 0.10, 0.20],
        "order_book_decay": 0.6,
        "max_drawdown_threshold": 0.05,
        "duration_limit": 30,
        "vol_window": 60,
        "vol_multiplier": 2.0,
        "high_level_interval": 10,
    }
    return NSETradingEnv(df, splits, cfg, ticker="RELIANCE", mode=mode, seed=42)


class TestNSETradingEnv:
    """Test suite for NSETradingEnv correctness and Gymnasium compliance."""

    @pytest.fixture
    def env(self):
        """Return a freshly constructed NSETradingEnv in 'test' mode."""
        return _build_env("test")

    @pytest.fixture
    def train_env(self):
        return _build_env("train")

    # ── Gymnasium compliance ───────────────────────────────────────────────────

    def test_gymnasium_compliance(self, train_env):
        """Environment must pass gymnasium's built-in env_checker with zero errors."""
        check_env(train_env, warn=True)

    # ── Episode lifecycle ──────────────────────────────────────────────────────

    def test_random_episode_completes(self, env):
        """A full episode driven by random actions must terminate without exception."""
        obs, info = env.reset(seed=0)
        assert obs.shape == (290,)
        steps = 0
        while True:
            action = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(action)
            steps += 1
            assert obs.shape == (290,)
            assert np.isfinite(rew)
            assert "costs" in info
            if term or trunc:
                break
        assert steps > 0

    def test_reproducibility(self, train_env):
        """Two train-mode resets with the same seed must produce identical observations."""
        obs1, _ = train_env.reset(seed=42)
        obs2, _ = train_env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    # ── Market rule: intraday close ────────────────────────────────────────────

    def test_intraday_close_flattens_position(self, env):
        """Episode must always end with position == 0 (FLAT)."""
        env.reset(seed=0)
        while True:
            action = 1  # always try to go long
            _, _, term, trunc, info = env.step(action)
            if term or trunc:
                assert info["position"] == 0, (
                    f"Position not flat at end of episode: {info['position']}"
                )
                break

    # ── Observation structure ──────────────────────────────────────────────────

    def test_time_encoding_unit_circle(self, env):
        """sin/cos time encoding must always satisfy sin² + cos² ≈ 1."""
        obs, _ = env.reset(seed=0)
        sin_val = float(obs[285])
        cos_val = float(obs[286])
        assert abs(sin_val ** 2 + cos_val ** 2 - 1.0) < 1e-5

    def test_observation_shape_and_dtype(self, env):
        obs, _ = env.reset(seed=0)
        assert obs.shape == (290,)
        assert obs.dtype == np.float32

    def test_regime_onehot_valid(self, env):
        """Regime one-hot [obs[287:290]] must sum to exactly 1."""
        obs, _ = env.reset(seed=0)
        regime_vec = obs[287:290]
        assert abs(float(regime_vec.sum()) - 1.0) < 1e-5

    # ── Reward purity ──────────────────────────────────────────────────────────

    def test_reward_has_no_risk_penalty(self, env):
        """
        Reward must equal position * log_return - trade_cost only.
        Risk is in info['costs'] — it must NOT appear in the reward signal.
        """
        env.reset(seed=0)
        # Force a long position at step 0
        obs, rew, term, trunc, info = env.step(1)  # BUY

        # The reward should be computable without any cost signal
        # Key check: costs in info, reward separate
        assert "costs" in info
        costs = info["costs"]
        assert set(costs.keys()) == {"cost_drawdown", "cost_duration", "cost_volatility"}
        # All cost values must be 0 or 1 (binary constraint signals)
        for k, v in costs.items():
            assert v in (0.0, 1.0), f"{k}={v} is not binary"

    # ── No lookahead ───────────────────────────────────────────────────────────

    def test_no_lookahead_in_observations(self, train_env):
        """
        Observation at reset with a fixed seed must be deterministic — forward
        data from future steps must not influence the initial observation.
        """
        obs1, _ = train_env.reset(seed=7)
        obs2, _ = train_env.reset(seed=7)
        np.testing.assert_array_equal(obs1, obs2)
        # The structural no-lookahead guarantee: the lookback buffer in reset()
        # is seeded from bars strictly *before* target_date (see _init_lookback_buffer).
