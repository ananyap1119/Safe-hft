"""
One-time Kite Connect OAuth login helper.

Run this script once before running data/fetch.py. It will:
  1. Print the Kite login URL — open it in your browser.
  2. After you log in, your browser will redirect to your app's redirect URL
     (e.g. https://localhost/?request_token=XXXXX&status=success).
  3. Paste the request_token from that URL when prompted.
  4. This script exchanges it for an access_token and writes it to .env.

Access tokens expire daily at midnight IST, so re-run this script if you need
to call the API on a different day. For our use case (one-time bulk download),
you only need to do this once.

Usage
-----
    cd safe_hft
    python scripts/kite_auth.py
"""

import os
import re
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

ENV_PATH = Path(__file__).parent.parent / ".env"


def main() -> None:
    load_dotenv(ENV_PATH)

    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")

    if not api_key or not api_secret:
        print("ERROR: KITE_API_KEY and KITE_API_SECRET must be set in .env")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)

    login_url = kite.login_url()
    print("\n" + "=" * 60)
    print("STEP 1: Open this URL in your browser and log in with your")
    print("        Zerodha credentials:")
    print()
    print(f"  {login_url}")
    print()
    print("STEP 2: After login, you will be redirected to a URL like:")
    print("  https://localhost/?request_token=XXXXXX&action=login&status=success")
    print()
    print("STEP 3: Copy the full redirect URL (or just the request_token value)")
    print("        and paste it below.")
    print("=" * 60 + "\n")

    raw_input = input("Paste the redirect URL or just the request_token: ").strip()

    # Accept either the full redirect URL or just the token string
    match = re.search(r"request_token=([A-Za-z0-9]+)", raw_input)
    if match:
        request_token = match.group(1)
    elif re.fullmatch(r"[A-Za-z0-9]+", raw_input):
        request_token = raw_input
    else:
        print("ERROR: Could not parse a request_token from your input.")
        sys.exit(1)

    print(f"\nExchanging request_token for access_token...")
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as exc:
        print(f"ERROR: Session generation failed: {exc}")
        sys.exit(1)

    access_token = data["access_token"]
    set_key(str(ENV_PATH), "KITE_ACCESS_TOKEN", access_token)

    print(f"\nSuccess! access_token written to .env")
    print(f"Token (first 8 chars): {access_token[:8]}...")
    print("\nYou can now run:  python -m data.fetch")


if __name__ == "__main__":
    main()
