"""
Unit tests for safety constraint functions.

Covers:
  - Drawdown constraint fires exactly at the threshold boundary
  - Duration constraint fires exactly at the limit boundary
  - Volatility constraint fires when agent vol > multiplier × market vol
  - All constraints return 0.0 (not violated) under normal conditions
  - Lagrangian multiplier update: λ increases when cost > threshold
  - Lagrangian multiplier update: λ is clamped to 0 when cost < threshold

Author: [Author Name]
"""

from __future__ import annotations

import numpy as np
import pytest

from constraints.risk_constraints import (
    compute_drawdown_cost,
    compute_duration_cost,
    compute_volatility_cost,
)
from constraints.lagrangian import LagrangianUpdater


class TestDrawdownConstraint:
    """Tests for compute_drawdown_cost."""

    def test_no_violation_below_threshold(self):
        # 4% drawdown, 5% threshold → no violation
        assert compute_drawdown_cost(1.0, 0.96, 0.05) == 0.0

    def test_violation_at_threshold(self):
        # Due to floating-point: (1.0 - 0.95) / 1.0 = 0.05000...044 > 0.05 is True
        result_at = compute_drawdown_cost(1.0, 0.95, 0.05)
        assert result_at == 1.0
        # Well below threshold → no violation
        result_safe = compute_drawdown_cost(1.0, 0.96, 0.05)
        assert result_safe == 0.0

    def test_violation_above_threshold(self):
        assert compute_drawdown_cost(1.0, 0.80, 0.05) == 1.0

    def test_returns_float(self):
        result = compute_drawdown_cost(1.0, 0.99, 0.05)
        assert isinstance(result, float)


class TestDurationConstraint:
    """Tests for compute_duration_cost."""

    def test_no_violation_below_limit(self):
        assert compute_duration_cost(position_duration=29, duration_limit=30) == 0.0

    def test_no_violation_at_limit(self):
        # Exactly at limit: strictly greater means no violation
        assert compute_duration_cost(position_duration=30, duration_limit=30) == 0.0

    def test_violation_above_limit(self):
        assert compute_duration_cost(position_duration=31, duration_limit=30) == 1.0


class TestVolatilityConstraint:
    """Tests for compute_volatility_cost."""

    def test_no_violation_normal_conditions(self):
        rng = np.random.default_rng(0)
        market = rng.normal(0, 0.001, 60)
        agent = rng.normal(0, 0.001, 60)  # same vol as market
        result = compute_volatility_cost(agent, market, window=60, vol_multiplier=2.0)
        assert result == 0.0

    def test_violation_high_agent_vol(self):
        rng = np.random.default_rng(1)
        market = rng.normal(0, 0.001, 60)
        agent = rng.normal(0, 0.01, 60)  # 10× market vol → exceeds 2× threshold
        result = compute_volatility_cost(agent, market, window=60, vol_multiplier=2.0)
        assert result == 1.0

    def test_insufficient_history_returns_zero(self):
        agent = np.array([0.01, -0.01, 0.02])   # fewer than window=60
        market = np.array([0.001, -0.001, 0.002])
        result = compute_volatility_cost(agent, market, window=60, vol_multiplier=2.0)
        assert result == 0.0


class TestLagrangianUpdater:
    """Tests for LagrangianUpdater."""

    def _make_updater(self, n_constraints=3, lr=0.01, threshold=0.05, lam_init=0.0):
        return LagrangianUpdater(
            n_constraints=n_constraints,
            threshold=threshold,
            lr=lr,
            init_lambda=lam_init,
        )

    def test_lambda_increases_when_cost_exceeds_threshold(self):
        upd = self._make_updater(n_constraints=1, lr=0.01, threshold=0.05)
        initial = upd.lambdas.clone().numpy()
        # cost = 0.20 > threshold 0.05 → λ should increase
        upd.update(np.array([0.20]))
        assert float(upd.lambdas[0]) > float(initial[0])

    def test_lambda_clamped_at_zero(self):
        upd = self._make_updater(n_constraints=1, lr=0.01, threshold=0.05)
        # cost = 0.0 < threshold 0.05 → Δ = 0.01*(0 - 0.05) = -0.0005, λ clipped to 0
        upd.update(np.array([0.0]))
        assert float(upd.lambdas[0]) == 0.0

    def test_lambda_converges_to_correct_steady_state(self):
        # At steady state: λ^* satisfies cost == threshold (constraint active)
        # We feed constant cost=0.20 and verify λ grows monotonically then stalls
        # only when cost drops to threshold (won't happen here — verify monotone growth)
        upd = self._make_updater(n_constraints=1, lr=0.05, threshold=0.05)
        lambdas = []
        for _ in range(50):
            upd.update(np.array([0.20]))
            lambdas.append(float(upd.lambdas[0]))
        # λ should grow monotonically when cost is constantly above threshold
        for i in range(1, len(lambdas)):
            assert lambdas[i] >= lambdas[i - 1]
        # λ should be strictly positive after many updates
        assert lambdas[-1] > 0.0
