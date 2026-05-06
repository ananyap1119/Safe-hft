"""
NSE-specific market microstructure rules for the SAFE-HFT simulation environment.

Implements:
  1. Tick-size rounding — Rs 0.05 per SEBI regulations.
  2. Circuit breakers   — SEBI-mandated intraday halts at +-5%, +-10%, +-20%.
  3. Intraday close rule — all positions must be flat by end of session
                           (agent action is overridden after 3:15 PM IST).

Author: [Author Name]
"""

from __future__ import annotations

import numpy as np

TICK_SIZE: float = 0.05
CIRCUIT_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.20)
INTRADAY_CLOSE_MINUTE: int = 360   # >= this minute, open positions are force-closed
TOTAL_SESSION_MINUTES: int = 375   # 9:15 AM to 3:30 PM IST


class NSEMarketRules:
    """Stateless collection of NSE market rule checkers and price adjusters."""

    @staticmethod
    def round_to_tick(price: float) -> float:
        """
        Round *price* to the nearest NSE tick size of Rs 0.05.

        Parameters
        ----------
        price : float

        Returns
        -------
        float
        """
        return round(round(price / TICK_SIZE) * TICK_SIZE, 2)

    @staticmethod
    def check_circuit_breaker(cumulative_daily_return: float) -> bool:
        """
        Return True if a circuit breaker is triggered.

        Parameters
        ----------
        cumulative_daily_return : float
            Cumulative log return since session open (e.g. 0.06 for +6%).

        Returns
        -------
        bool
        """
        abs_ret = abs(cumulative_daily_return)
        return any(abs_ret >= t for t in CIRCUIT_THRESHOLDS)

    @staticmethod
    def apply_circuit_breaker(action: int, cumulative_daily_return: float) -> int:
        """
        Override the agent's action to Hold (0) if a circuit breaker is active.

        Parameters
        ----------
        action : int
        cumulative_daily_return : float

        Returns
        -------
        int
        """
        if NSEMarketRules.check_circuit_breaker(cumulative_daily_return):
            return 0
        return action

    @staticmethod
    def apply_intraday_close(action: int, minute_index: int, position: int) -> int:
        """
        Force-close any open position at or after minute 360 (3:15 PM IST).

        Parameters
        ----------
        action : int
        minute_index : int
            0 = 9:15 AM, 374 = 3:29 PM.
        position : int
            -1 (short), 0 (flat), 1 (long).

        Returns
        -------
        int
            3 (Close) if forced close required, else original *action*.
        """
        if minute_index >= INTRADAY_CLOSE_MINUTE and position != 0:
            return 3
        return action

    @staticmethod
    def compute_transaction_cost(trade_value: float, cost_rate: float = 0.0003) -> float:
        """
        Compute one-way transaction cost.

        Parameters
        ----------
        trade_value : float
            Absolute notional value (entry or exit).
        cost_rate : float
            Default 0.0003 (0.03%).

        Returns
        -------
        float
            Cost as a fraction of trade value.
        """
        return cost_rate

    @staticmethod
    def compute_slippage(spread_proxy: float, slippage_rate: float = 0.0001) -> float:
        """
        Estimate slippage as a fraction of the spread proxy.

        Parameters
        ----------
        spread_proxy : float
            Normalised (high - low) / close for the current bar.
        slippage_rate : float
            Default 0.0001 (0.01%).

        Returns
        -------
        float
            Slippage cost as a fraction of trade value.
        """
        return slippage_rate * max(spread_proxy, 0.0)
