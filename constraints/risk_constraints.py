"""
Safety constraint definitions for SAFE-HFT.

Each function returns 1.0 (violated) or 0.0 (satisfied). These binary cost signals
are returned in ``info["costs"]`` from the environment and consumed by OmniSafe's
Lagrangian multiplier update. They are NEVER added to the reward signal.

Author: [Author Name]
"""

from __future__ import annotations

import numpy as np


def compute_drawdown_cost(
    peak_portfolio_value: float,
    current_portfolio_value: float,
    threshold: float = 0.05,
) -> float:
    """
    Evaluate the maximum drawdown safety constraint.

    Parameters
    ----------
    peak_portfolio_value : float
    current_portfolio_value : float
    threshold : float
        Violation if drawdown exceeds this fraction of peak (default: 0.05).

    Returns
    -------
    float
        1.0 if violated, 0.0 otherwise.
    """
    if peak_portfolio_value <= 0:
        return 0.0
    drawdown = (peak_portfolio_value - current_portfolio_value) / peak_portfolio_value
    return 1.0 if drawdown > threshold else 0.0


def compute_duration_cost(
    position_duration: int,
    duration_limit: int = 30,
) -> float:
    """
    Evaluate the position duration safety constraint.

    Parameters
    ----------
    position_duration : int
        Consecutive bars in current directional position.
    duration_limit : int
        Violation if duration strictly exceeds this limit (default: 30).

    Returns
    -------
    float
        1.0 if violated, 0.0 otherwise.
    """
    return 1.0 if position_duration > duration_limit else 0.0


def compute_volatility_cost(
    agent_returns: np.ndarray,
    market_returns: np.ndarray,
    window: int = 60,
    vol_multiplier: float = 2.0,
) -> float:
    """
    Evaluate the realised volatility safety constraint.

    Parameters
    ----------
    agent_returns : np.ndarray
        Per-bar PnL returns for the agent (recent ``window`` bars or fewer).
    market_returns : np.ndarray
        Per-bar log returns of the underlying stock (same window).
    window : int
        Minimum history required before the constraint can fire (default: 60).
    vol_multiplier : float
        Violation if agent_vol > multiplier * market_vol (default: 2.0).

    Returns
    -------
    float
        1.0 if violated, 0.0 otherwise. Returns 0.0 when history < window.
    """
    if len(agent_returns) < window or len(market_returns) < 2:
        return 0.0

    agent_vol = float(np.std(agent_returns[-window:], ddof=1))
    market_vol = float(np.std(market_returns, ddof=1))

    if market_vol == 0.0:
        return 0.0
    return 1.0 if agent_vol > vol_multiplier * market_vol else 0.0


def compute_total_cost(
    peak_portfolio_value: float,
    current_portfolio_value: float,
    position_duration: int,
    agent_returns: np.ndarray,
    market_returns: np.ndarray,
    thresholds: dict,
) -> tuple[float, float, float]:
    """
    Convenience wrapper: evaluate all three constraints.

    Parameters
    ----------
    thresholds : dict
        Keys: ``max_drawdown_threshold``, ``duration_limit``,
        ``vol_multiplier``, ``vol_window``.

    Returns
    -------
    tuple[float, float, float]
        ``(cost_drawdown, cost_duration, cost_volatility)``.
    """
    c_dd = compute_drawdown_cost(
        peak_portfolio_value,
        current_portfolio_value,
        threshold=float(thresholds.get("max_drawdown_threshold", 0.05)),
    )
    c_dur = compute_duration_cost(
        position_duration,
        duration_limit=int(thresholds.get("duration_limit", 30)),
    )
    c_vol = compute_volatility_cost(
        agent_returns,
        market_returns,
        window=int(thresholds.get("vol_window", 60)),
        vol_multiplier=float(thresholds.get("vol_multiplier", 2.0)),
    )
    return c_dd, c_dur, c_vol
