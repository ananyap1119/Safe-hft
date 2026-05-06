"""
Simplified three-level order book simulation derived from OHLCV data.

All values are pre-computed during the preprocessing phase and stored in the
processed parquet files; this module is called once at preprocessing time, not
at runtime during RL training (no per-step overhead).

Derivation:
  best_bid  = close - (high - low) / 4
  best_ask  = close + (high - low) / 4
  spread    = (high - low) / 2

Author: [Author Name]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SimulatedOrderBook:
    """
    Pre-compute order book proxy features from a processed OHLCV DataFrame.

    Parameters
    ----------
    decay : float
        Fraction of bar volume assigned to the best level (level 1).
        Levels 2 and 3 get ``decay*(1-decay)`` and ``(1-decay)**2``.
        Default: 0.6.
    """

    def __init__(self, decay: float = 0.6) -> None:
        self.decay = decay

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append order book proxy columns to *df* and return the augmented copy.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with ``high``, ``low``, ``close``, ``volume``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with additional columns:
            ``best_bid``, ``best_ask``, ``mid_price``, ``bid_ask_spread``,
            ``vol_level1``, ``vol_level2``, ``vol_level3``.
        """
        df = df.copy()
        half_spread = (df["high"] - df["low"]) / 4.0
        df["best_bid"] = df["close"] - half_spread
        df["best_ask"] = df["close"] + half_spread
        df["mid_price"] = df["close"]
        df["bid_ask_spread"] = (df["high"] - df["low"]) / 2.0

        vol_l1, vol_l2, vol_l3 = self._split_volume(df["volume"].astype(float))
        df["vol_level1"] = vol_l1
        df["vol_level2"] = vol_l2
        df["vol_level3"] = vol_l3
        return df

    def get_bid_ask(self, bar: pd.Series) -> tuple[float, float]:
        """
        Return ``(best_bid, best_ask)`` for a single bar.

        Parameters
        ----------
        bar : pd.Series
            Row from an OHLCV DataFrame with ``high``, ``low``, ``close``.

        Returns
        -------
        tuple[float, float]
        """
        half_spread = (bar["high"] - bar["low"]) / 4.0
        return float(bar["close"] - half_spread), float(bar["close"] + half_spread)

    def compute_volume_levels(self, volume: float) -> tuple[float, float, float]:
        """
        Split bar volume across three simulated order book levels.

        Parameters
        ----------
        volume : float

        Returns
        -------
        tuple[float, float, float]
            (level1, level2, level3) summing to ``volume``.
        """
        l1 = self.decay * volume
        l2 = self.decay * (1 - self.decay) * volume
        l3 = volume - l1 - l2
        return l1, l2, l3

    def _split_volume(
        self, volume: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        l1 = self.decay * volume
        l2 = self.decay * (1 - self.decay) * volume
        l3 = volume - l1 - l2
        return l1, l2, l3
