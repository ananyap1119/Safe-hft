"""
Held-out test set evaluation for all SAFE-HFT agents.

Loads the best checkpoint for each agent (selected by val-set Sharpe ratio during
training) and runs a single deterministic episode on the test split. Reports the full
metric suite required for the paper's Table 1.

Metrics reported per agent
---------------------------
  Annualised Return (%), Sharpe Ratio, Sortino Ratio, Calmar Ratio,
  Maximum Drawdown (%), Win Rate (%),
  Constraint violation rate (%), Drawdown violations (%),
  Duration violations (%), Volatility violations (%),
  Average trade duration (bars), Total trades

Usage
-----
    python -m train.evaluate --config train/config.yaml --agent all
    python -m train.evaluate --config train/config.yaml --agent safe_hft --ticker RELIANCE

Author: [Author Name]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "datasets" / "processed"

AGENTS: list[str] = ["safe_hft", "ppo", "dqn", "macrohft", "earnhft", "random"]
TICKERS: list[str] = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

# Metrics where *higher* is better (used when bolding the LaTeX table)
HIGHER_IS_BETTER: set[str] = {
    "annualised_return_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "win_rate_pct",
}
# Metrics where *lower* is better
LOWER_IS_BETTER: set[str] = {
    "max_drawdown_pct",
    "constraint_violation_pct",
    "drawdown_violation_pct",
    "duration_violation_pct",
    "volatility_violation_pct",
}

COLUMN_DISPLAY: dict[str, str] = {
    "annualised_return_pct": r"Ann. Return (\%)",
    "sharpe_ratio":          "Sharpe",
    "sortino_ratio":         "Sortino",
    "calmar_ratio":          "Calmar",
    "max_drawdown_pct":      r"Max DD (\%)",
    "win_rate_pct":          r"Win Rate (\%)",
    "constraint_violation_pct":   r"Viol. (\%)",
    "drawdown_violation_pct":     r"DD Viol. (\%)",
    "duration_violation_pct":     r"Dur. Viol. (\%)",
    "volatility_violation_pct":   r"Vol. Viol. (\%)",
    "avg_trade_duration_bars":    "Avg Dur (bars)",
    "total_trades":               "N Trades",
}

COLUMN_DECIMALS: dict[str, int] = {
    "annualised_return_pct": 2,
    "sharpe_ratio":          3,
    "sortino_ratio":         3,
    "calmar_ratio":          3,
    "max_drawdown_pct":      2,
    "win_rate_pct":          1,
    "constraint_violation_pct":   1,
    "drawdown_violation_pct":     1,
    "duration_violation_pct":     1,
    "volatility_violation_pct":   1,
    "avg_trade_duration_bars":    1,
    "total_trades":               0,
}


# ── Evaluation core ───────────────────────────────────────────────────────────

def evaluate_agent(
    agent: Any,
    env: Any,
    config: dict,
    agent_name: str,
) -> dict[str, float]:
    """
    Run one full deterministic pass over all test days and compute all metrics.

    Iterates until env._day_cursor has exhausted every test day. For MacroHFT
    the regime is read from env._current_regime to dispatch to the correct
    sub-agent.

    Parameters
    ----------
    agent : Any
        Trained agent with a ``predict(obs, *, deterministic)`` method.
        MacroHFT additionally accepts a ``regime`` keyword argument.
    env : Any
        ``NSETradingEnv`` initialised in ``"test"`` mode.
    config : dict
    agent_name : str

    Returns
    -------
    dict[str, float]  full metric suite for Table 1
    """
    from metrics.finance_metrics import compute_all_metrics

    returns: list[float] = []
    trade_pnls: list[float] = []
    cost_dd: list[float] = []
    cost_dur: list[float] = []
    cost_vol: list[float] = []
    trade_durations: list[float] = []

    # Trade tracking state
    in_trade: bool = False
    pv_at_entry: float = 1.0
    trade_entry_step: int = 0
    total_steps: int = 0

    obs, _ = env.reset(seed=0)

    while True:
        # Action selection — MacroHFT needs regime
        if agent_name == "macrohft":
            regime = int(env._current_regime) if hasattr(env, "_current_regime") else 2
            action = agent.predict(obs, regime=regime, deterministic=True)
        elif hasattr(agent, "predict"):
            action = agent.predict(obs, deterministic=True)
        else:
            action = int(env.action_space.sample())

        obs, rew, term, trunc, info = env.step(action)
        returns.append(float(rew))
        total_steps += 1

        # Cost signals
        costs = info.get("costs", {})
        cost_dd.append(float(costs.get("cost_drawdown", 0.0)))
        cost_dur.append(float(costs.get("cost_duration", 0.0)))
        cost_vol.append(float(costs.get("cost_volatility", 0.0)))

        # Trade PnL / duration tracking
        pos = info.get("position", 0)
        pv = float(info.get("portfolio_value", 1.0))
        trade = bool(info.get("trade", False))

        if trade:
            if in_trade:
                # Close the previous position
                trade_pnls.append(pv - pv_at_entry)
                trade_durations.append(float(total_steps - trade_entry_step))
                in_trade = False
            if pos != 0:
                # Open a new position
                in_trade = True
                pv_at_entry = pv
                trade_entry_step = total_steps

        if term or trunc:
            # Force-close any open trade at episode end
            if in_trade:
                trade_pnls.append(pv - pv_at_entry)
                trade_durations.append(float(total_steps - trade_entry_step))
                in_trade = False

            # Advance to the next test day if any remain
            if hasattr(env, "_day_cursor") and env._day_cursor < env._n_days:
                obs, _ = env.reset()
            else:
                break

    return compute_all_metrics(
        returns=np.array(returns, dtype=np.float64),
        trade_pnls=np.array(trade_pnls, dtype=np.float64),
        cost_drawdown=np.array(cost_dd, dtype=np.float64),
        cost_duration=np.array(cost_dur, dtype=np.float64),
        cost_volatility=np.array(cost_vol, dtype=np.float64),
        trade_durations=np.array(trade_durations, dtype=np.float64),
    )


def compute_buy_and_hold(data: pd.DataFrame, initial_capital: float = 1.0) -> dict[str, float]:
    """
    Compute buy-and-hold benchmark metrics analytically from test-set price data.

    Uses raw close prices to derive per-bar log-returns. A single long position
    is held from bar 0 to bar N; cost signals are all zero (no constraints).

    Parameters
    ----------
    data : pd.DataFrame
        Test-split processed DataFrame with a ``close`` column at original scale.
    initial_capital : float
        Starting portfolio value (default 1.0).

    Returns
    -------
    dict[str, float]  same metric keys as ``evaluate_agent``
    """
    from metrics.finance_metrics import compute_all_metrics

    close = data["close"].dropna().values.astype(np.float64)
    if len(close) < 2:
        logger.warning("Insufficient close prices for buy-and-hold benchmark.")
        return {k: float("nan") for k in COLUMN_DISPLAY}

    returns = np.log(close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan))
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return {k: float("nan") for k in COLUMN_DISPLAY}

    total_pnl = float(np.prod(1.0 + returns) - 1.0) * initial_capital
    zeros = np.zeros(len(returns), dtype=np.float64)

    return compute_all_metrics(
        returns=returns,
        trade_pnls=np.array([total_pnl], dtype=np.float64),
        cost_drawdown=zeros,
        cost_duration=zeros,
        cost_volatility=zeros,
        trade_durations=np.array([float(len(returns))], dtype=np.float64),
    )


# ── Table construction ────────────────────────────────────────────────────────

def build_results_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Assemble per-agent metric dicts into a comparison DataFrame.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Outer key: agent display name. Inner dict: metric key → value.

    Returns
    -------
    pd.DataFrame  rows=agents, columns=metrics
    """
    rows = []
    for agent_name, metrics in results.items():
        row = {"agent": agent_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("agent")
    # Keep only known columns in a fixed display order
    ordered_cols = [c for c in COLUMN_DISPLAY if c in df.columns]
    return df[ordered_cols]


def save_latex_table(df: pd.DataFrame, output_path: Path) -> None:
    """
    Write a LaTeX-formatted comparison table to *output_path*.

    Best value in each metric column is wrapped in ``\\textbf{}``.
    Table uses ``booktabs`` rules and one decimal-aligned column per metric.

    Parameters
    ----------
    df : pd.DataFrame  results table from ``build_results_table``
    output_path : Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_cols = len(df.columns)
    col_spec = "l" + "r" * n_cols

    header_line = " & ".join(
        COLUMN_DISPLAY.get(c, c) for c in df.columns
    )

    def _best_idx(col: str, series: pd.Series) -> str | None:
        """Return the index label of the best finite value, or None."""
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            return None
        if col in HIGHER_IS_BETTER:
            return str(finite.idxmax())
        if col in LOWER_IS_BETTER:
            return str(finite.idxmin())
        return None

    best: dict[str, str | None] = {c: _best_idx(c, df[c]) for c in df.columns}

    def _fmt(val: float, col: str) -> str:
        dp = COLUMN_DECIMALS.get(col, 2)
        if not np.isfinite(val):
            return r"--"
        return f"{val:.{dp}f}"

    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Test-set performance comparison (RELIANCE, 2023-07-18 to 2023-12-29).}",
        r"\label{tab:results}",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"Agent & " + header_line + r" \\",
        r"\midrule",
    ]

    safe_hft_drawn = False
    for agent in df.index:
        # Draw a midrule before baselines to separate primary from baselines
        if not safe_hft_drawn and agent != "SAFE-HFT":
            safe_hft_drawn = True
            lines.append(r"\midrule")
        elif safe_hft_drawn and agent == r"Buy \& Hold":
            lines.append(r"\midrule")

        cells = [agent]
        for col in df.columns:
            val = df.loc[agent, col]
            fmted = _fmt(float(val), col)
            if best[col] is not None and str(agent) == best[col]:
                fmted = r"\textbf{" + fmted + r"}"
            cells.append(fmted)
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("LaTeX table written to %s", output_path)


# ── Checkpoint loading ────────────────────────────────────────────────────────

def _load_env(ticker: str, config: dict) -> "NSETradingEnv":
    """Build a test-mode NSETradingEnv for *ticker*."""
    from env.trading_env import NSETradingEnv

    df_path = DATA_DIR / f"{ticker}_processed.parquet"
    split_path = DATA_DIR / f"{ticker}_splits.json"
    if not df_path.exists():
        raise FileNotFoundError(f"No processed data for {ticker}. Run data/preprocess.py first.")
    df = pd.read_parquet(df_path)
    with open(split_path, encoding="utf-8") as f:
        splits = json.load(f)

    env_cfg: dict = {}
    env_cfg.update(config.get("constraints", {}))
    env_cfg.update(config.get("env", {}))
    return NSETradingEnv(df, splits, env_cfg, ticker=ticker, mode="test",
                         seed=config["agent"]["seed"])


def _load_agent(agent_name: str, env: Any, config: dict, ckpt_dir: Path) -> Any:
    """
    Instantiate and restore an agent from *ckpt_dir*.

    Parameters
    ----------
    agent_name : str
    env : NSETradingEnv   test environment (needed to re-create SB3 wrappers)
    config : dict
    ckpt_dir : Path       e.g. ``checkpoints/safe_hft/RELIANCE/``

    Returns
    -------
    Agent object with a ``predict(obs, *, deterministic)`` method.
    """
    seed = config["agent"]["seed"]

    if agent_name == "safe_hft":
        from agents.safe_hft_agent import SafeHFTAgent
        device = config["agent"].get("device", "cpu")
        agent = SafeHFTAgent(config, device=device, seed=seed)
        best_ckpt = ckpt_dir / "best.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(
                f"No best.pt checkpoint for safe_hft/{env.ticker}. "
                "Run train.train first."
            )
        agent.load(best_ckpt)
        return agent

    if agent_name == "ppo":
        from agents.ppo_baseline import PPOBaseline
        agent = PPOBaseline(env, config, seed=seed)
        ckpt = ckpt_dir / "ppo_final"
        if not (ckpt_dir / "ppo_final.zip").exists():
            raise FileNotFoundError(f"No ppo_final.zip in {ckpt_dir}")
        agent.load(ckpt)
        return agent

    if agent_name == "dqn":
        from agents.dqn_baseline import DQNBaseline
        agent = DQNBaseline(env, config, seed=seed)
        ckpt = ckpt_dir / "dqn_final"
        if not (ckpt_dir / "dqn_final.zip").exists():
            raise FileNotFoundError(f"No dqn_final.zip in {ckpt_dir}")
        agent.load(ckpt)
        return agent

    if agent_name == "macrohft":
        from agents.macrohft_baseline import MacroHFTBaseline
        agent = MacroHFTBaseline(env, env, env, config, seed=seed)
        sub_a = ckpt_dir / "macrohft_sub_a.zip"
        if not sub_a.exists():
            raise FileNotFoundError(f"No macrohft_sub_a.zip in {ckpt_dir}")
        agent.load(ckpt_dir / "macrohft")
        return agent

    if agent_name == "earnhft":
        from agents.earnhft_baseline import EarnHFTBaseline
        agent = EarnHFTBaseline(env, config, seed=seed)
        hl_ckpt = ckpt_dir / "earnhft_hl.zip"
        if not hl_ckpt.exists():
            raise FileNotFoundError(f"No earnhft_hl.zip in {ckpt_dir}")
        agent.load(ckpt_dir / "earnhft")
        return agent

    if agent_name == "random":
        from agents.random_baseline import RandomBaseline
        return RandomBaseline(n_actions=4, seed=seed)

    raise ValueError(f"Unknown agent: {agent_name}")


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI args, evaluate all requested agents/tickers, save outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Evaluate SAFE-HFT agents on test set")
    parser.add_argument("--config", default="train/config.yaml")
    parser.add_argument("--agent", default="all",
                        choices=AGENTS + ["all"])
    parser.add_argument("--ticker", default="RELIANCE",
                        choices=TICKERS + ["all"])
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    agents_to_eval = AGENTS if args.agent == "all" else [args.agent]
    tickers_to_eval = TICKERS if args.ticker == "all" else [args.ticker]

    ckpt_root = ROOT / args.checkpoint_dir
    out_root = ROOT / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    # Display names for the table
    agent_display: dict[str, str] = {
        "safe_hft": "SAFE-HFT",
        "ppo":      "PPO",
        "dqn":      "DQN",
        "macrohft": "MacroHFT",
        "earnhft":  "EarnHFT",
        "random":   "Random",
    }

    for ticker in tickers_to_eval:
        logger.info("=" * 60)
        logger.info("Evaluating on ticker: %s", ticker)
        logger.info("=" * 60)

        results: dict[str, dict[str, float]] = {}

        # Load test env data for B&H benchmark (loaded once)
        df_path = DATA_DIR / f"{ticker}_processed.parquet"
        split_path = DATA_DIR / f"{ticker}_splits.json"
        if df_path.exists():
            df_full = pd.read_parquet(df_path)
            with open(split_path, encoding="utf-8") as f:
                splits = json.load(f)
            test_start = pd.Timestamp(splits["test"]["start"])
            test_end = pd.Timestamp(splits["test"]["end"])
            test_df = df_full.loc[test_start:test_end]
        else:
            test_df = pd.DataFrame()

        for agent_name in agents_to_eval:
            display = agent_display.get(agent_name, agent_name)
            logger.info("  Evaluating %s ...", display)
            try:
                env = _load_env(ticker, config)
                ckpt_dir = ckpt_root / agent_name / ticker
                agent = _load_agent(agent_name, env, config, ckpt_dir)
                metrics = evaluate_agent(agent, env, config, agent_name)
                results[display] = metrics
                logger.info(
                    "    Sharpe=%.3f  MaxDD=%.2f%%  Viol=%.1f%%",
                    metrics.get("sharpe_ratio", float("nan")),
                    metrics.get("max_drawdown_pct", float("nan")),
                    metrics.get("constraint_violation_pct", float("nan")),
                )
            except FileNotFoundError as exc:
                logger.warning("  Skipping %s — checkpoint not found: %s", display, exc)
            except Exception as exc:
                logger.error("  FAILED %s — %s", display, exc, exc_info=True)

        # Buy-and-hold benchmark
        if not test_df.empty and "close" in test_df.columns:
            logger.info("  Computing Buy-and-Hold benchmark ...")
            bh_metrics = compute_buy_and_hold(test_df)
            results[r"Buy \& Hold"] = bh_metrics
            logger.info(
                "    Sharpe=%.3f  MaxDD=%.2f%%",
                bh_metrics.get("sharpe_ratio", float("nan")),
                bh_metrics.get("max_drawdown_pct", float("nan")),
            )

        if not results:
            logger.warning("No results for %s — skipping table generation.", ticker)
            continue

        # Build and save table
        table = build_results_table(results)

        csv_path = out_root / f"results_{ticker}.csv"
        table.to_csv(csv_path)
        logger.info("CSV saved to %s", csv_path)

        tex_path = out_root / f"table_{ticker}.tex"
        save_latex_table(table, tex_path)

        # Also print a human-readable summary
        logger.info("\n%s", table.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
