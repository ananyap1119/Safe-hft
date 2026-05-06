"""
Feature engineering and normalisation pipeline for SAFE-HFT.

Computes all technical indicators and microstructure proxies from raw OHLCV data.
All computations are strictly causal (no lookahead bias): each bar uses only data
available at that bar's close timestamp.

Features computed
-----------------
Technical indicators:
  log_return, volatility_10, volatility_30, volatility_60,
  rsi_14, macd, macd_signal, macd_hist, bb_width_20, volume_ratio_30

Microstructure proxies (derived from OHLCV):
  spread_proxy, price_impact, volume_imbalance, ob_imbalance

Regime label (from data/regime_classifier.py):
  regime  (0 = trending-up, 1 = trending-down, 2 = high-volatility)

All numeric features (excluding regime) are rolling-z-score normalised over a
60-bar rolling window before being saved to the processed dataset.

Author: [Author Name]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data.regime_classifier import RegimeClassifier

logger = logging.getLogger(__name__)

# All feature columns in order (regime is last; it is NOT z-score normalised)
FEATURE_COLUMNS: list[str] = [
    "log_return",
    "volatility_10",
    "volatility_30",
    "volatility_60",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width_20",
    "volume_ratio_30",
    "spread_proxy",
    "price_impact",
    "volume_imbalance",
    "ob_imbalance",
    "regime",
]

FEATURES_TO_NORMALIZE: list[str] = [c for c in FEATURE_COLUMNS if c != "regime"]


class DataPreprocessor:
    """
    Transform raw OHLCV parquet files into feature-rich, normalised datasets.

    Parameters
    ----------
    raw_dir : Path
        Directory containing ``<TICKER>_raw.parquet`` files.
    processed_dir : Path
        Output directory for ``<TICKER>_processed.parquet`` files.
    norm_window : int
        Rolling window size for z-score normalisation (default: 60).
    regime_window : int
        Look-back window for regime classification (default: 20).
    return_threshold : float
        Return threshold for regime labelling (default: 0.005).
    train_frac : float
        Fraction of time periods used for training (default: 0.70).
    val_frac : float
        Fraction of time periods used for validation (default: 0.15).
    """

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        norm_window: int = 60,
        regime_window: int = 20,
        return_threshold: float = 0.005,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.norm_window = norm_window
        self.regime_window = regime_window
        self.return_threshold = return_threshold
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def process_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Run the full feature-engineering and normalisation pipeline for one ticker.

        Steps
        -----
        1. Load raw parquet.
        2. Compute technical indicators.
        3. Compute microstructure features.
        4. Fit the regime classifier on the training split only, then classify all bars.
        5. Apply rolling z-score normalisation to all numeric features.
        6. Drop rows where any feature is NaN (initial look-back warm-up period).
        7. Save processed parquet.
        8. Save train/val/test split indices as JSON.

        Parameters
        ----------
        ticker : str
            NSE ticker symbol.

        Returns
        -------
        pd.DataFrame
            Fully processed DataFrame.
        """
        raw_path = self.raw_dir / f"{ticker}_raw.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw data not found: {raw_path}. Run data/fetch.py first."
            )

        logger.info("%s: loading raw data...", ticker)
        df = pd.read_parquet(raw_path)

        logger.info("%s: computing technical indicators...", ticker)
        df = self.compute_technical_indicators(df)

        logger.info("%s: computing microstructure features...", ticker)
        df = self.compute_microstructure_features(df)

        # Determine the training date boundary BEFORE fitting the regime classifier
        # to avoid any lookahead from the fit step.
        dates = df.index.normalize().unique().sort_values()
        n_train_days = int(len(dates) * self.train_frac)
        train_cutoff = dates[n_train_days - 1]
        train_mask = df.index.normalize() <= train_cutoff

        logger.info(
            "%s: fitting regime classifier on training data "
            "(up to %s, %d bars)...",
            ticker,
            train_cutoff.date(),
            train_mask.sum(),
        )
        clf = RegimeClassifier(
            window=self.regime_window,
            return_threshold=self.return_threshold,
        )
        clf.fit(df.loc[train_mask])
        df["regime"] = clf.classify(df)

        logger.info(
            "%s: vol_threshold = %.6f", ticker, clf._fitted_vol_threshold
        )
        _log_regime_balance(df, ticker)

        logger.info("%s: applying rolling z-score normalisation...", ticker)
        for col in FEATURES_TO_NORMALIZE:
            df[col] = self.rolling_zscore(df[col], self.norm_window)

        # Drop warm-up rows where any feature is still NaN
        n_before = len(df)
        df = df.dropna(subset=FEATURE_COLUMNS)
        n_dropped = n_before - len(df)
        logger.info(
            "%s: dropped %d warm-up rows (%.1f days), %d bars remain.",
            ticker,
            n_dropped,
            n_dropped / 375,
            len(df),
        )

        # Reorder to OHLCV + features
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        df = df[ohlcv_cols + FEATURE_COLUMNS]

        out_path = self.processed_dir / f"{ticker}_processed.parquet"
        df.to_parquet(out_path, index=True, compression="snappy")
        logger.info("%s: saved to %s", ticker, out_path.name)

        self.save_split_indices(df, ticker, self.train_frac, self.val_frac)

        return df

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append technical-indicator columns to *df* (returns a copy).

        All indicators are computed with strictly past data only (no centre=True).

        Columns added
        -------------
        log_return, volatility_10, volatility_30, volatility_60,
        rsi_14, macd, macd_signal, macd_hist, bb_width_20, volume_ratio_30

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame with columns ``[open, high, low, close, volume]``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame augmented with indicator columns.
        """
        df = df.copy()
        close = df["close"]
        volume = df["volume"].astype("float64")

        # Log return
        df["log_return"] = np.log(close / close.shift(1))

        # Rolling volatility
        for w in (10, 30, 60):
            df[f"volatility_{w}"] = (
                df["log_return"].rolling(w, min_periods=w).std(ddof=1)
            )

        # RSI-14 (Wilder's smoothing: EMA with alpha = 1/14, adjust=False)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        alpha = 1.0 / 14
        avg_gain = gain.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
        ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = macd_line - signal_line

        # Bollinger Band width (normalised by SMA so it's scale-invariant)
        sma20 = close.rolling(20, min_periods=20).mean()
        std20 = close.rolling(20, min_periods=20).std(ddof=1)
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20
        df["bb_width_20"] = (bb_upper - bb_lower) / sma20.replace(0.0, np.nan)

        # Volume ratio: current bar volume / 30-bar rolling mean
        vol_ma30 = volume.rolling(30, min_periods=30).mean()
        df["volume_ratio_30"] = volume / vol_ma30.replace(0.0, np.nan)

        return df

    def compute_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append OHLCV-derived microstructure proxy columns to *df* (returns a copy).

        Computed columns
        ----------------
        spread_proxy     : (high - low) / close
        price_impact     : |log_return| / volume_ratio_30  (proxy for market impact)
        volume_imbalance : (vol_t - vol_{t-1}) / (vol_t + vol_{t-1})
        ob_imbalance     : sign(close - open) * volume_ratio_30

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that already contains ``log_return`` and ``volume_ratio_30``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame augmented with microstructure columns.
        """
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]
        volume = df["volume"].astype("float64")

        # Bid-ask spread proxy: high-low range normalised by close
        df["spread_proxy"] = (high - low) / close.replace(0.0, np.nan)

        # Price impact: absolute return per unit of volume pressure
        vol_ratio = df["volume_ratio_30"]
        df["price_impact"] = df["log_return"].abs() / vol_ratio.replace(0.0, np.nan)

        # Volume imbalance: signed relative change in volume vs previous bar
        vol_prev = volume.shift(1)
        vol_sum = volume + vol_prev
        df["volume_imbalance"] = (volume - vol_prev) / vol_sum.replace(0.0, np.nan)

        # Order book imbalance proxy: direction × volume pressure
        direction = np.sign(close - open_)
        df["ob_imbalance"] = direction * vol_ratio

        return df

    def rolling_zscore(self, series: pd.Series, window: int) -> pd.Series:
        """
        Apply a causal rolling z-score normalisation to *series*.

        Parameters
        ----------
        series : pd.Series
            Raw feature series.
        window : int
            Look-back window for computing mean and std.

        Returns
        -------
        pd.Series
            Normalised series (NaN for the first ``window - 1`` entries).
            Rows where rolling std == 0 are set to 0.0 (constant segment).
        """
        roll = series.rolling(window=window, min_periods=window)
        mu = roll.mean()
        sigma = roll.std(ddof=1)
        normalised = (series - mu) / sigma.replace(0.0, np.nan)
        # Fill divisions by zero with 0 (constant-valued window)
        return normalised.fillna(0.0).where(series.notna())

    def save_split_indices(
        self,
        df: pd.DataFrame,
        ticker: str,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
    ) -> dict[str, dict[str, str]]:
        """
        Compute and save chronological train/val/test date boundaries.

        Splits are made by unique trading date — shuffling is never applied
        to financial time series as it introduces lookahead bias.

        Parameters
        ----------
        df : pd.DataFrame
            Processed DataFrame with DatetimeIndex.
        ticker : str
            Used to name the output JSON file.
        train_frac : float
        val_frac : float

        Returns
        -------
        dict
            ``{"train": {"start": ..., "end": ...}, "val": ..., "test": ...}``
        """
        # Keep as pandas DatetimeIndex so elements support .date()
        dates = df.index.normalize().unique().sort_values()
        n = len(dates)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_dates = dates[:n_train]
        val_dates = dates[n_train : n_train + n_val]
        test_dates = dates[n_train + n_val :]

        splits = {
            "train": {
                "start": str(train_dates[0].date()),
                "end": str(train_dates[-1].date()),
                "n_days": int(len(train_dates)),
            },
            "val": {
                "start": str(val_dates[0].date()),
                "end": str(val_dates[-1].date()),
                "n_days": int(len(val_dates)),
            },
            "test": {
                "start": str(test_dates[0].date()),
                "end": str(test_dates[-1].date()),
                "n_days": int(len(test_dates)),
            },
        }

        out_path = self.processed_dir / f"{ticker}_splits.json"
        with open(out_path, "w") as f:
            json.dump(splits, f, indent=2)

        logger.info(
            "%s splits: train %s→%s (%d days)  val %s→%s (%d days)  "
            "test %s→%s (%d days)",
            ticker,
            splits["train"]["start"], splits["train"]["end"], splits["train"]["n_days"],
            splits["val"]["start"],   splits["val"]["end"],   splits["val"]["n_days"],
            splits["test"]["start"],  splits["test"]["end"],  splits["test"]["n_days"],
        )
        return splits


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_regime_balance(df: pd.DataFrame, ticker: str) -> None:
    """Log the fraction of bars in each regime (diagnostic only)."""
    counts = df["regime"].value_counts(dropna=True).sort_index()
    total = counts.sum()
    labels = {0: "trending_up", 1: "trending_down", 2: "high_vol"}
    parts = [
        f"{labels.get(int(k), k)}={v/total:.1%}"
        for k, v in counts.items()
    ]
    logger.info("%s regime balance: %s", ticker, "  ".join(parts))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Run the preprocessing pipeline for all five tickers."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from data.fetch import TICKERS, RAW_DIR
    PROCESSED_DIR = RAW_DIR.parent / "processed"

    parser = argparse.ArgumentParser(description="Preprocess NSE OHLCV data")
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    args = parser.parse_args()

    preprocessor = DataPreprocessor(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
    )

    results: dict[str, int] = {}
    for ticker in args.tickers:
        try:
            df = preprocessor.process_ticker(ticker)
            results[ticker] = len(df)
        except Exception as exc:
            logger.error("%s: FAILED — %s", ticker, exc)

    print()
    print("-" * 52)
    total = 0
    for ticker, n in results.items():
        print(f"  {ticker:<12}  {n:>8,} bars after preprocessing")
        total += n
    print(f"  {'TOTAL':<12}  {total:>8,} bars")
    print("-" * 52)
    print(f"\nProcessed files: {Path(args.processed_dir).resolve()}")


if __name__ == "__main__":
    main()
