"""
Financial performance metrics for SAFE-HFT evaluation.

All functions are stateless and operate on numpy arrays of per-bar returns
or per-trade PnL values.

Author: [Author Name]
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR: int = 252
BARS_PER_DAY: int = 375
BARS_PER_YEAR: int = TRADING_DAYS_PER_YEAR * BARS_PER_DAY


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    returns : np.ndarray  per-bar portfolio returns
    risk_free_rate : float  annual rate

    Returns
    -------
    float  np.nan if std == 0
    """
    if len(returns) == 0:
        return np.nan
    rf_per_bar = risk_free_rate / BARS_PER_YEAR
    excess = returns - rf_per_bar
    std = np.std(excess, ddof=1)
    if std == 0.0:
        return np.nan
    return float(np.mean(excess) / std * np.sqrt(BARS_PER_YEAR))


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sortino ratio (downside deviation only).

    Parameters
    ----------
    returns : np.ndarray
    risk_free_rate : float  annual

    Returns
    -------
    float
    """
    if len(returns) == 0:
        return np.nan
    rf_per_bar = risk_free_rate / BARS_PER_YEAR
    excess = returns - rf_per_bar
    downside = excess[excess < 0.0]
    if len(downside) == 0:
        return np.inf
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0.0:
        return np.nan
    return float(np.mean(excess) / downside_std * np.sqrt(BARS_PER_YEAR))


def max_drawdown(returns: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown as a positive fraction.

    Parameters
    ----------
    returns : np.ndarray  per-bar returns

    Returns
    -------
    float  in [0, 1], e.g. 0.12 = 12% drawdown
    """
    if len(returns) == 0:
        return 0.0
    cum = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd))


def annualised_return(returns: np.ndarray) -> float:
    """
    Annualised portfolio return.

    Parameters
    ----------
    returns : np.ndarray  per-bar returns

    Returns
    -------
    float  fraction, e.g. 0.15 = 15%/year
    """
    if len(returns) == 0:
        return 0.0
    total = float(np.prod(1.0 + returns))
    n_years = len(returns) / BARS_PER_YEAR
    if n_years <= 0:
        return 0.0
    return float(total ** (1.0 / n_years) - 1.0)


def calmar_ratio(returns: np.ndarray) -> float:
    """
    Calmar ratio: annualised return / max drawdown.

    Parameters
    ----------
    returns : np.ndarray

    Returns
    -------
    float  np.nan if max_dd == 0
    """
    ann_ret = annualised_return(returns)
    mdd = max_drawdown(returns)
    if mdd == 0.0:
        return np.nan
    return float(ann_ret / mdd)


def win_rate(trade_pnls: np.ndarray) -> float:
    """
    Fraction of closed trades that were profitable.

    Parameters
    ----------
    trade_pnls : np.ndarray  per-trade PnL values

    Returns
    -------
    float  in [0, 1], np.nan if no trades
    """
    if len(trade_pnls) == 0:
        return np.nan
    return float(np.sum(trade_pnls > 0) / len(trade_pnls))


def constraint_violation_rate(cost_signals: np.ndarray) -> float:
    """
    Fraction of timesteps where the given constraint was violated.

    Parameters
    ----------
    cost_signals : np.ndarray  binary (0 or 1)

    Returns
    -------
    float  in [0, 1]
    """
    if len(cost_signals) == 0:
        return 0.0
    return float(np.mean(cost_signals))


def compute_all_metrics(
    returns: np.ndarray,
    trade_pnls: np.ndarray,
    cost_drawdown: np.ndarray,
    cost_duration: np.ndarray,
    cost_volatility: np.ndarray,
    trade_durations: np.ndarray,
) -> dict[str, float]:
    """
    Compute the full metric suite required for Table 1 of the paper.

    Parameters
    ----------
    returns : np.ndarray  per-bar portfolio returns
    trade_pnls : np.ndarray  per-trade PnL
    cost_drawdown : np.ndarray  binary violation signal per bar
    cost_duration : np.ndarray  binary violation signal per bar
    cost_volatility : np.ndarray  binary violation signal per bar
    trade_durations : np.ndarray  bars per completed trade

    Returns
    -------
    dict[str, float]
    """
    any_cost = np.clip(cost_drawdown + cost_duration + cost_volatility, 0.0, 1.0)
    return {
        "annualised_return_pct": annualised_return(returns) * 100.0,
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "calmar_ratio": calmar_ratio(returns),
        "max_drawdown_pct": max_drawdown(returns) * 100.0,
        "win_rate_pct": win_rate(trade_pnls) * 100.0,
        "constraint_violation_pct": constraint_violation_rate(any_cost) * 100.0,
        "drawdown_violation_pct": constraint_violation_rate(cost_drawdown) * 100.0,
        "duration_violation_pct": constraint_violation_rate(cost_duration) * 100.0,
        "volatility_violation_pct": constraint_violation_rate(cost_volatility) * 100.0,
        "avg_trade_duration_bars": float(np.mean(trade_durations)) if len(trade_durations) > 0 else 0.0,
        "total_trades": float(len(trade_pnls)),
    }
