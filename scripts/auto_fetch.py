"""
Full automation: Kite OAuth login + NSE data download in one command.

What this script does automatically:
  1. Starts a local HTTP server on port 5000 to capture the OAuth redirect.
  2. Opens your browser to the Kite login page.
  3. Waits for you to log in (the ONLY manual step).
  4. Captures the request_token from the redirect automatically.
  5. Exchanges it for an access_token and saves it to .env.
  6. Runs the full NSE data download for all 5 tickers (2021-2023).

Prerequisites
-------------
  - In Kite developer portal, set your app's Redirect URL to:
        http://127.0.0.1:5000
  - pip install kiteconnect python-dotenv (already in requirements.txt)

Usage
-----
    cd C:\\Users\\anany\\safe_hft
    python scripts/auto_fetch.py
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

from data.fetch import NSEDataFetcher, TICKERS, DATE_START, DATE_END, RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).parent.parent / ".env"
CALLBACK_PORT = 5000
CALLBACK_HOST = "127.0.0.1"

# Shared state between the HTTP handler and the main thread
_captured_token: dict[str, str] = {}
_server_ready = threading.Event()
_token_received = threading.Event()


class _CallbackHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that captures the request_token from Kite's redirect."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "request_token" in params and "error" not in params:
            _captured_token["value"] = params["request_token"][0]
            _token_received.set()
            self._respond(
                200,
                "<html><body style='font-family:sans-serif;padding:40px'>"
                "<h2 style='color:#4CAF50'>Login successful!</h2>"
                "<p>You can close this tab and return to the terminal.</p>"
                "</body></html>",
            )
        elif "error" in params:
            error = params.get("error", ["unknown"])[0]
            self._respond(
                400,
                f"<html><body style='font-family:sans-serif;padding:40px'>"
                f"<h2 style='color:#f44336'>Login failed</h2>"
                f"<p>Error: {error}</p>"
                f"</body></html>",
            )
        else:
            self._respond(404, "Not found")

    def _respond(self, code: int, body: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format: str, *args) -> None:
        pass  # suppress default access logs


def _run_server(server: HTTPServer) -> None:
    _server_ready.set()
    server.serve_forever()


def get_access_token(api_key: str, api_secret: str) -> str:
    """
    Start local OAuth callback server, open browser, wait for token.

    Parameters
    ----------
    api_key : str
    api_secret : str

    Returns
    -------
    str
        A valid Kite access_token.
    """
    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    # Start the callback server in a background thread
    server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), _CallbackHandler)
    thread = threading.Thread(target=_run_server, args=(server,), daemon=True)
    thread.start()
    _server_ready.wait(timeout=3)

    logger.info("Opening Kite login in your browser...")
    logger.info("Please log in with your Zerodha credentials.")
    webbrowser.open(login_url)

    print("\n" + "=" * 58)
    print("  Browser opened. Log in to Zerodha in the browser window.")
    print("  This terminal will continue automatically after login.")
    print("=" * 58 + "\n")

    received = _token_received.wait(timeout=300)  # 5-minute window
    server.shutdown()

    if not received or "value" not in _captured_token:
        raise TimeoutError(
            "No login callback received within 5 minutes. "
            "Make sure your app's Redirect URL is set to "
            f"http://{CALLBACK_HOST}:{CALLBACK_PORT}"
        )

    request_token = _captured_token["value"]
    logger.info("request_token captured. Exchanging for access_token...")

    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token: str = data["access_token"]

    set_key(str(ENV_PATH), "KITE_ACCESS_TOKEN", access_token)
    logger.info("access_token saved to .env")

    return access_token


def main() -> None:
    load_dotenv(ENV_PATH)

    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")

    if not api_key or not api_secret:
        logger.error("KITE_API_KEY / KITE_API_SECRET not found in .env")
        sys.exit(1)

    # ── Step 1: OAuth login ───────────────────────────────────────────────────
    print("\n" + "━" * 58)
    print("  SAFE-HFT  —  NSE Data Downloader")
    print("━" * 58)
    print("  Step 1/2: Kite Connect login")
    print("━" * 58 + "\n")

    try:
        access_token = get_access_token(api_key, api_secret)
    except TimeoutError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.error("Login failed: %s", exc)
        logger.error(
            "Make sure your Kite app Redirect URL is exactly:  "
            "http://%s:%d", CALLBACK_HOST, CALLBACK_PORT
        )
        sys.exit(1)

    print("\n  Login successful!\n")

    # Inject the fresh token into the current process's environment so that
    # NSEDataFetcher.connect() picks it up without needing a second load_dotenv.
    os.environ["KITE_ACCESS_TOKEN"] = access_token

    # ── Step 2: Data download ─────────────────────────────────────────────────
    print("━" * 58)
    print("  Step 2/2: Downloading NSE minute-level data")
    print(f"  Tickers : {', '.join(TICKERS)}")
    print(f"  Range   : {DATE_START}  →  {DATE_END}")
    print(f"  Output  : {RAW_DIR.resolve()}")
    print("━" * 58 + "\n")

    fetcher = NSEDataFetcher()
    saved = fetcher.fetch_all()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "━" * 58)
    print("  Download complete")
    print("━" * 58)

    import pandas as pd
    total_bars = 0
    for ticker, path in saved.items():
        df = pd.read_parquet(path)
        total_bars += len(df)
        date_range = f"{df.index.min().date()}  →  {df.index.max().date()}"
        print(f"  {ticker:<12}  {len(df):>8,} bars   {date_range}")

    print(f"\n  Total: {total_bars:,} bars across {len(saved)} tickers")
    print(f"\n  Next step: run the preprocessing pipeline")
    print(f"    python -m data.preprocess")
    print("━" * 58 + "\n")


if __name__ == "__main__":
    main()
