"""
Run all 6 agents on all 5 tickers sequentially with per-agent log files.

Usage
-----
    # All agents, all tickers (30 jobs, ~28 hrs on CPU):
    python scripts/train_all.py

    # One ticker, all agents (~5.7 hrs):
    python scripts/train_all.py --ticker RELIANCE

    # One agent, all tickers (~5 hrs):
    python scripts/train_all.py --agent safe_hft

    # Skip already-completed checkpoints:
    python scripts/train_all.py --skip-existing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

AGENTS  = ["safe_hft", "ppo", "dqn", "macrohft", "earnhft", "random"]
TICKERS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

CKPT_SENTINEL = {
    "safe_hft": "best.pt",
    "ppo":      "ppo_final.zip",
    "dqn":      "dqn_final.zip",
    "macrohft": "macrohft_sub_a.zip",
    "earnhft":  "earnhft_ll.zip",
    "random":   None,   # no checkpoint needed
}


def is_done(agent: str, ticker: str) -> bool:
    sentinel = CKPT_SENTINEL.get(agent)
    if sentinel is None:
        return True   # random needs no training
    return (ROOT / "checkpoints" / agent / ticker / sentinel).exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",  default="all", choices=AGENTS + ["all"])
    parser.add_argument("--ticker", default="all", choices=TICKERS + ["all"])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    agents  = AGENTS  if args.agent  == "all" else [args.agent]
    tickers = TICKERS if args.ticker == "all" else [args.ticker]

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    total = len(agents) * len(tickers)
    done_count = 0

    for ticker in tickers:
        for agent in agents:
            done_count += 1
            tag = f"{agent}/{ticker}"

            if args.skip_existing and is_done(agent, ticker):
                print(f"[{done_count}/{total}] SKIP (checkpoint exists): {tag}")
                continue

            log_path = log_dir / f"{agent}_{ticker}.log"
            print(f"[{done_count}/{total}] Starting {tag}  -> {log_path}", flush=True)

            with open(log_path, "w") as f:
                result = subprocess.run(
                    [sys.executable, "-m", "train.train",
                     "--agent", agent,
                     "--ticker", ticker],
                    cwd=ROOT,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )

            if result.returncode != 0:
                print(f"  FAILED (exit {result.returncode}) — see {log_path}")
            else:
                print(f"  OK")

    print("All training jobs complete.")


if __name__ == "__main__":
    main()
