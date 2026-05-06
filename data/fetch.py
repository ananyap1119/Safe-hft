"""
NSE historical data fetcher for SAFE-HFT.

Downloads minute-level OHLCV data for five Nifty 50 stocks (RELIANCE, TCS, INFY,
HDFCBANK, ICICIBANK) for 2021-01-01 to 2023-12-31 via Kite Connect's historical
data API. Saves each ticker as a parquet file in data/datasets/raw/.

Kite API constraints
--------------------
- Minute-interval data: maximum 60 calendar days per request.
- Requests are split into 60-day chunks with a 0.5s delay between calls.
- Instrument tokens are resolved dynamically from kite.instruments("NSE").

Authentication
--------------
Run scripts/kite_auth.py once before running this module. It writes
KITE_ACCESS_TOKEN to .env.

Usage
-----
    cd safe_hft
    python -m data.fetch                          # fetch all tickers
    python -m data.fetch --tickers RELIANCE TCS   # fetch specific tickers

Author: [Author Name]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TICKERS: list[str] = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
DATE_START: str = "2021-01-01"
DATE_END: str = "2023-12-31"
INTERVAL: str = "minute"
CHUNK_DAYS: int = 60        # Kite's max window for minute-level requests
REQUEST_DELAY: float = 0.5  # seconds between API calls (respect rate limits)

ENV_PATH: Path = Path(__file__).parent.parent / ".env"
RAW_DIR: Path = Path(__file__).parent / "datasets" / "raw"

# Expected OHLCV columns returned by Kite
KITE_COLUMNS: list[str] = ["date", "open", "high", "low", "close", "volume"]

# NSE trading session (IST): 9:15 AM to 3:30 PM
SESSION_START_HOUR: int = 9
SESSION_START_MINUTE: int = 15
SESSION_END_HOUR: int = 15
SESSION_END_MINUTE: int = 30


# ── Main class ────────────────────────────────────────────────────────────────

class NSEDataFetcher:
    """
    Fetch and persist raw minute-level OHLCV data from NSE via Kite Connect.

    Parameters
    ----------
    output_dir : Path
        Directory where raw parquet files will be written.
    tickers : list[str], optional
        NSE ticker symbols to fetch. Defaults to the five Nifty 50 stocks.
    start_date : str
        ISO-format start date, inclusive (e.g. ``"2021-01-01"``).
    end_date : str
        ISO-format end date, inclusive (e.g. ``"2023-12-31"``).
    """

    def __init__(
        self,
        output_dir: Path = RAW_DIR,
        tickers: Optional[list[str]] = None,
        start_date: str = DATE_START,
        end_date: str = DATE_END,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.tickers = tickers or TICKERS
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._kite: Optional[KiteConnect] = None
        self._token_map: dict[str, int] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Initialise the KiteConnect client using credentials from .env.

        Raises
        ------
        EnvironmentError
            If KITE_API_KEY or KITE_ACCESS_TOKEN are missing from the environment.
        """
        load_dotenv(ENV_PATH)
        api_key = os.getenv("KITE_API_KEY")
        access_token = os.getenv("KITE_ACCESS_TOKEN")

        if not api_key:
            raise EnvironmentError("KITE_API_KEY not found in .env")
        if not access_token:
            raise EnvironmentError(
                "KITE_ACCESS_TOKEN not found in .env. "
                "Run scripts/kite_auth.py first."
            )

        self._kite = KiteConnect(api_key=api_key)
        self._kite.set_access_token(access_token)
        logger.info("Kite Connect client initialised (api_key: %s...)", api_key[:8])

    def fetch_all(self) -> dict[str, Path]:
        """
        Fetch data for all configured tickers and save to parquet.

        Returns
        -------
        dict[str, Path]
            Mapping of ticker symbol to saved parquet file path.
        """
        self._ensure_connected()
        self._build_token_map()

        saved: dict[str, Path] = {}
        for ticker in self.tickers:
            logger.info("Fetching %s ...", ticker)
            try:
                df = self.fetch_ticker(ticker)
                path = self.save_raw(ticker, df)
                saved[ticker] = path
                logger.info(
                    "  %s: %d bars saved to %s", ticker, len(df), path.name
                )
            except Exception as exc:
                logger.error("  %s: FAILED — %s", ticker, exc)
        return saved

    def fetch_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Fetch the full date-range of minute bars for a single ticker.

        Splits the range into 60-day chunks and merges them.

        Parameters
        ----------
        ticker : str
            NSE ticker symbol (e.g. ``"RELIANCE"``).

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex (IST, tz-naive) and columns
            ``[open, high, low, close, volume]``. Only bars within the
            NSE trading session (9:15–15:30 IST) are retained.

        Raises
        ------
        ValueError
            If the ticker is not found in the NSE instrument list.
        RuntimeError
            If all chunk fetches fail.
        """
        self._ensure_connected()
        token = self._resolve_token(ticker)
        chunks: list[pd.DataFrame] = []
        failed_chunks: int = 0

        for chunk_start, chunk_end in self._date_chunks(
            self.start_date, self.end_date, CHUNK_DAYS
        ):
            logger.debug(
                "  %s: fetching %s → %s", ticker, chunk_start, chunk_end
            )
            try:
                raw = self._kite.historical_data(
                    instrument_token=token,
                    from_date=chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                    to_date=chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                    interval=INTERVAL,
                    continuous=False,
                    oi=False,
                )
            except Exception as exc:
                logger.warning(
                    "  %s chunk %s–%s failed: %s", ticker, chunk_start, chunk_end, exc
                )
                failed_chunks += 1
                time.sleep(REQUEST_DELAY * 4)
                continue

            if raw:
                df_chunk = pd.DataFrame(raw, columns=KITE_COLUMNS)
                chunks.append(df_chunk)

            time.sleep(REQUEST_DELAY)

        if not chunks:
            raise RuntimeError(
                f"All {failed_chunks} chunks failed for {ticker}. "
                "Check access_token validity."
            )

        df = pd.concat(chunks, ignore_index=True)
        df = self._clean(df, ticker)
        self._validate_ohlcv(df, ticker)
        return df

    def save_raw(self, ticker: str, df: pd.DataFrame) -> Path:
        """
        Persist a raw OHLCV DataFrame to ``output_dir/<ticker>_raw.parquet``.

        Parameters
        ----------
        ticker : str
            Ticker symbol; used as filename stem.
        df : pd.DataFrame
            Cleaned OHLCV DataFrame with DatetimeIndex.

        Returns
        -------
        Path
            Absolute path to the written parquet file.
        """
        path = self.output_dir / f"{ticker}_raw.parquet"
        df.to_parquet(path, index=True, compression="snappy")
        return path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self._kite is None:
            self.connect()

    def _build_token_map(self) -> None:
        """
        Download the full NSE instrument list and build a ticker → token map.

        Raises
        ------
        RuntimeError
            If any of the configured tickers are not found.
        """
        logger.info("Loading NSE instrument list from Kite...")
        instruments = self._kite.instruments("NSE")
        instrument_df = pd.DataFrame(instruments)

        # Kite returns segment-qualified symbols; equity instruments use
        # tradingsymbol directly
        for ticker in self.tickers:
            match = instrument_df[
                (instrument_df["tradingsymbol"] == ticker)
                & (instrument_df["segment"] == "NSE")
                & (instrument_df["instrument_type"] == "EQ")
            ]
            if match.empty:
                raise RuntimeError(
                    f"Ticker '{ticker}' not found in NSE instrument list. "
                    "Check the symbol spelling."
                )
            self._token_map[ticker] = int(match.iloc[0]["instrument_token"])
            logger.debug("  %s → token %d", ticker, self._token_map[ticker])

        logger.info("Instrument tokens resolved for %d tickers.", len(self._token_map))

    def _resolve_token(self, ticker: str) -> int:
        if not self._token_map:
            self._build_token_map()
        if ticker not in self._token_map:
            raise ValueError(
                f"Token for '{ticker}' not available. "
                f"Known tickers: {list(self._token_map.keys())}"
            )
        return self._token_map[ticker]

    def _clean(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Standardise column names, set DatetimeIndex, filter to session hours,
        and drop duplicates.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame from Kite with a ``date`` column.
        ticker : str
            Used only for logging.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with a timezone-naive DatetimeIndex (IST).
        """
        df = df.copy()

        # Kite returns datetime objects; convert to pandas Timestamp
        df["date"] = pd.to_datetime(df["date"])

        # Strip timezone info — Kite returns IST-aware; we work tz-naive
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)

        df = df.set_index("date").sort_index()
        df.index.name = "datetime"

        # Keep only market hours: 9:15 to 15:30
        df = df.between_time(
            f"{SESSION_START_HOUR:02d}:{SESSION_START_MINUTE:02d}",
            f"{SESSION_END_HOUR:02d}:{SESSION_END_MINUTE:02d}",
        )

        # Drop exact duplicate timestamps (can occur at chunk boundaries)
        df = df[~df.index.duplicated(keep="first")]

        # Ensure correct dtypes
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype("float64")
        df["volume"] = df["volume"].astype("int64")

        n_before = len(df)
        df = df.dropna(subset=["open", "high", "low", "close"])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.warning("  %s: dropped %d rows with NaN prices", ticker, n_dropped)

        return df

    def _validate_ohlcv(self, df: pd.DataFrame, ticker: str) -> None:
        """
        Assert that the DataFrame has the expected schema and reasonable content.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned OHLCV DataFrame.
        ticker : str
            Used in error messages.

        Raises
        ------
        ValueError
            If required columns are missing, DataFrame is empty, or price
            sanity checks fail (e.g. high < low).
        """
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{ticker}: missing columns {missing}")

        if df.empty:
            raise ValueError(f"{ticker}: resulting DataFrame is empty")

        if (df["high"] < df["low"]).any():
            n_bad = (df["high"] < df["low"]).sum()
            raise ValueError(f"{ticker}: {n_bad} bars have high < low")

        if (df["close"] <= 0).any():
            n_bad = (df["close"] <= 0).sum()
            raise ValueError(f"{ticker}: {n_bad} bars have close <= 0")

        if (df["volume"] < 0).any():
            raise ValueError(f"{ticker}: negative volume values found")

        expected_cols_order = ["open", "high", "low", "close", "volume"]
        df = df[expected_cols_order]  # reorder but don't raise

        logger.info(
            "  %s validated: %d bars, %s to %s",
            ticker,
            len(df),
            df.index.min().date(),
            df.index.max().date(),
        )

    @staticmethod
    def _date_chunks(
        start: date, end: date, chunk_days: int
    ) -> list[tuple[date, date]]:
        """
        Split ``[start, end]`` into non-overlapping chunks of at most
        ``chunk_days`` calendar days each.

        Parameters
        ----------
        start : date
        end : date
        chunk_days : int

        Returns
        -------
        list[tuple[date, date]]
            List of (chunk_start, chunk_end) pairs.
        """
        chunks: list[tuple[date, date]] = []
        current = start
        delta = timedelta(days=chunk_days)
        while current <= end:
            chunk_end = min(current + delta - timedelta(days=1), end)
            chunks.append((current, chunk_end))
            current = chunk_end + timedelta(days=1)
        return chunks


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the fetch pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Fetch NSE minute-level OHLCV data")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=TICKERS,
        help="NSE ticker symbols to fetch (default: all 5)",
    )
    parser.add_argument(
        "--start", default=DATE_START, help="Start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", default=DATE_END, help="End date YYYY-MM-DD"
    )
    parser.add_argument(
        "--output-dir", default=str(RAW_DIR), help="Raw data output directory"
    )
    args = parser.parse_args()

    fetcher = NSEDataFetcher(
        output_dir=Path(args.output_dir),
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
    )

    saved = fetcher.fetch_all()

    print("\n── Fetch complete ──────────────────────────────────")
    for ticker, path in saved.items():
        df = pd.read_parquet(path)
        print(f"  {ticker:12s}  {len(df):>8,} bars  →  {path.name}")
    print(f"\nAll files in: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
