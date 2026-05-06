"""
Market regime classifier for SAFE-HFT state space construction.

Assigns one of three mutually-exclusive regime labels to every bar:
  0 — Trending up    (20-bar return >  +0.5%  AND  20-bar vol <= threshold)
  1 — Trending down  (20-bar return < -0.5%   AND  20-bar vol <= threshold)
  2 — High volatility (20-bar vol > threshold, regardless of direction)

The regime label is injected directly into the Gymnasium observation vector so that
the agent can adapt its behaviour to prevailing market conditions. This design is
inspired by MacroHFT (Zong et al., KDD 2024), which uses regime decomposition to
train specialised sub-agents. Unlike MacroHFT, we use regime as a state signal rather
than as a training-data partition.

Author: [Author Name]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_LABELS: dict[int, str] = {
    0: "trending_up",
    1: "trending_down",
    2: "high_volatility",
}


class RegimeClassifier:
    """
    Rule-based market regime classifier operating on a minute-OHLCV DataFrame.

    Parameters
    ----------
    window : int
        Look-back window (in bars) for computing rolling return and volatility.
        Default: 20.
    return_threshold : float
        Minimum absolute 20-bar return (as a fraction) to qualify as trending.
        Default: 0.005 (0.5%).
    vol_threshold : float or None
        Rolling volatility above which a bar is labelled high-volatility.
        Set ``None`` to auto-compute as the 75th percentile of the training
        distribution during ``fit()``.
    """

    def __init__(
        self,
        window: int = 20,
        return_threshold: float = 0.005,
        vol_threshold: float | None = None,
    ) -> None:
        self.window = window
        self.return_threshold = return_threshold
        self.vol_threshold = vol_threshold
        self._fitted_vol_threshold: float | None = vol_threshold

    def fit(self, df: pd.DataFrame) -> "RegimeClassifier":
        """
        Compute the volatility threshold from training-set statistics.

        Must be called only on training-split data to prevent lookahead bias.

        Parameters
        ----------
        df : pd.DataFrame
            Training-split OHLCV DataFrame with a ``log_return`` column.

        Returns
        -------
        RegimeClassifier
            Self, for method chaining.
        """
        if "log_return" not in df.columns:
            raise ValueError("DataFrame must have a 'log_return' column.")
        rolling_vol = self._rolling_volatility(df["log_return"])
        # 75th percentile of the training distribution
        self._fitted_vol_threshold = float(rolling_vol.quantile(0.75))
        return self

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """
        Assign a regime label to every bar in *df*.

        Uses only data up to and including each bar (fully causal).

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with a ``log_return`` column and a DatetimeIndex.

        Returns
        -------
        pd.Series
            Integer series (0, 1, or 2) aligned to *df*'s index. Leading
            ``window - 1`` values are NaN.

        Raises
        ------
        RuntimeError
            If neither ``vol_threshold`` was set manually nor ``fit()`` was called.
        """
        if self._fitted_vol_threshold is None:
            raise RuntimeError(
                "vol_threshold is unknown. Call fit() on training data first, "
                "or pass vol_threshold explicitly in the constructor."
            )
        if "log_return" not in df.columns:
            raise ValueError("DataFrame must have a 'log_return' column.")

        threshold = self._fitted_vol_threshold
        rolling_ret = self._rolling_return(df["log_return"])
        rolling_vol = self._rolling_volatility(df["log_return"])

        # Priority: check vol first, then direction
        regime = pd.Series(np.nan, index=df.index, dtype="float64")

        valid = rolling_ret.notna() & rolling_vol.notna()

        # Default: high volatility (also catches low-vol sideways as fallback)
        regime[valid] = 2

        # Override with directional labels where vol is low enough
        low_vol = valid & (rolling_vol <= threshold)
        regime[low_vol & (rolling_ret > self.return_threshold)] = 0
        regime[low_vol & (rolling_ret < -self.return_threshold)] = 1

        return regime.astype("Int64")  # nullable integer, preserves NaN

    def _rolling_return(self, log_return: pd.Series) -> pd.Series:
        """
        Compute the rolling sum of log returns over ``self.window`` bars.

        Parameters
        ----------
        log_return : pd.Series
            Per-bar log return series.

        Returns
        -------
        pd.Series
            Rolling window return; NaN for the first ``window - 1`` entries.
        """
        return log_return.rolling(window=self.window, min_periods=self.window).sum()

    def _rolling_volatility(self, log_return: pd.Series) -> pd.Series:
        """
        Compute the rolling standard deviation of log returns.

        Parameters
        ----------
        log_return : pd.Series
            Per-bar log return series.

        Returns
        -------
        pd.Series
            Rolling std (ddof=1); NaN for the first ``window - 1`` entries.
        """
        return log_return.rolling(
            window=self.window, min_periods=self.window
        ).std(ddof=1)
