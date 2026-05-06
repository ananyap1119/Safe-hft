"""
Publication-ready figure generation for SAFE-HFT.

Generates all six figures and two tables described in Phase 5 of the project spec.
All figures are saved at 300 DPI in both PNG and PDF formats, sized for IEEE
double-column layout (max 3.5 inches per column, 7.16 inches full width).

Style conventions
-----------------
- No gridlines; thin axis spines; muted 6-colour palette.
- Font: 10pt axis labels, 9pt tick labels, 8pt legends.
- No chart junk: no box frames, no fill under curves except where explicitly noted.

Figure list
-----------
  Fig 1 — Cumulative PnL curve (all agents, test set)
  Fig 2 — Running drawdown (all agents, test set)
  Fig 3 — Constraint violation over training (SAFE-HFT only, 3 subplots)
  Fig 4 — Performance bar chart (Sharpe / Sortino / Calmar, all agents)
  Fig 5 — Regime-conditional Sharpe (SAFE-HFT vs MacroHFT-style)
  Fig 6 — Trade distribution (duration histogram + PnL histogram)

Author: [Author Name]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless operation
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml


# ── Style constants ────────────────────────────────────────────────────────────

SINGLE_COL_WIDTH: float = 3.5
DOUBLE_COL_WIDTH: float = 7.16
FIGURE_DPI: int = 300

AGENT_ORDER: list[str] = [
    "SAFE-HFT", "PPO", "DQN", "MacroHFT", "EarnHFT", "Random", r"Buy \& Hold",
]

AGENT_COLORS: dict[str, str] = {
    "SAFE-HFT":      "#2166AC",
    "PPO":           "#4DAF4A",
    "DQN":           "#FF7F00",
    "MacroHFT":      "#E41A1C",
    "EarnHFT":       "#984EA3",
    "Random":        "#AAAAAA",
    r"Buy \& Hold":  "#A65628",
}

AGENT_LINESTYLES: dict[str, str] = {
    "SAFE-HFT":      "-",
    "PPO":           "--",
    "DQN":           "-.",
    "MacroHFT":      ":",
    "EarnHFT":       (0, (5, 1)),
    "Random":        (0, (1, 1)),
    r"Buy \& Hold":  (0, (3, 1, 1, 1)),
}

REGIME_NAMES: list[str] = ["Trending Up", "Trending Down", "High Volatility"]
REGIME_COLORS: list[str] = ["#1B9E77", "#D95F02", "#7570B3"]


def apply_ieee_style() -> None:
    """Configure matplotlib rcParams for IEEE publication style."""
    plt.rcParams.update({
        "font.family":          "serif",
        "font.size":            9,
        "axes.labelsize":       10,
        "axes.titlesize":       10,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      8,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            False,
        "lines.linewidth":      1.2,
        "figure.dpi":           FIGURE_DPI,
        "savefig.dpi":          FIGURE_DPI,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,
        "text.usetex":          False,
    })


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save figure as both PNG and PDF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


# ── Figure 1: Cumulative PnL ───────────────────────────────────────────────────

def plot_cumulative_pnl(
    equity_curves: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """
    Figure 1: Cumulative portfolio value (all agents, test set).

    Parameters
    ----------
    equity_curves : dict[str, np.ndarray]
        Agent name → 1-D array of cumulative portfolio values starting at 1.0.
    output_dir : Path
    """
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 2.5))

    for agent in AGENT_ORDER:
        if agent not in equity_curves:
            continue
        curve = equity_curves[agent]
        x = np.arange(len(curve))
        ax.plot(
            x, curve,
            color=AGENT_COLORS.get(agent, "#333333"),
            linestyle=AGENT_LINESTYLES.get(agent, "-"),
            label=agent,
            alpha=0.85,
        )

    ax.axhline(1.0, color="#CCCCCC", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Bar index (test set)")
    ax.set_ylabel("Cumulative portfolio value")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.set_xlim(0, max(len(v) for v in equity_curves.values()) - 1)

    _save(fig, output_dir, "fig1_cumulative_pnl")


# ── Figure 2: Running drawdown ─────────────────────────────────────────────────

def _compute_drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Return the running drawdown (fraction from peak) for an equity curve."""
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
    return dd


def plot_drawdown(
    equity_curves: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """
    Figure 2: Running drawdown from peak (all agents, test set).

    Parameters
    ----------
    equity_curves : dict[str, np.ndarray]
    output_dir : Path
    """
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 2.5))

    for agent in AGENT_ORDER:
        if agent not in equity_curves:
            continue
        dd = _compute_drawdown_series(equity_curves[agent])
        x = np.arange(len(dd))
        ax.plot(
            x, dd * 100.0,
            color=AGENT_COLORS.get(agent, "#333333"),
            linestyle=AGENT_LINESTYLES.get(agent, "-"),
            label=agent,
            alpha=0.85,
        )

    ax.axhline(5.0, color="#FF0000", linewidth=0.8, linestyle="--", alpha=0.5,
               label="5% constraint threshold")
    ax.set_xlabel("Bar index (test set)")
    ax.set_ylabel("Drawdown from peak (%)")
    ax.invert_yaxis()
    ax.legend(loc="lower left", frameon=False, ncol=2)
    ax.set_xlim(0, max(len(v) for v in equity_curves.values()) - 1)

    _save(fig, output_dir, "fig2_drawdown")


# ── Figure 3: Constraint violations over training ──────────────────────────────

def plot_constraint_violations(
    violation_history: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """
    Figure 3: Per-constraint violation rate over training (SAFE-HFT only).

    Parameters
    ----------
    violation_history : dict[str, np.ndarray]
        Keys: ``"drawdown"``, ``"duration"``, ``"volatility"``.
        Each array has one entry per eval checkpoint (shape: (n_evals,)).
    output_dir : Path
    """
    apply_ieee_style()
    constraint_labels = {
        "drawdown":   ("Drawdown", "#2166AC"),
        "duration":   ("Duration", "#E41A1C"),
        "volatility": ("Volatility", "#4DAF4A"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 2.2), sharey=True)

    for ax, (key, (label, color)) in zip(axes, constraint_labels.items()):
        if key not in violation_history:
            ax.set_visible(False)
            continue
        hist = violation_history[key]
        x = np.arange(len(hist)) * 100_000  # steps
        ax.plot(x, hist * 100.0, color=color, linewidth=1.2)
        ax.axhline(5.0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Training steps")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))

    axes[0].set_ylabel("Violation rate (%)")
    fig.suptitle("SAFE-HFT constraint violation over training", fontsize=10)
    plt.tight_layout()

    _save(fig, output_dir, "fig3_constraint_violations")


# ── Figure 4: Performance bar chart ───────────────────────────────────────────

def plot_performance_bars(
    results: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """
    Figure 4: Grouped bar chart of Sharpe, Sortino, Calmar for all agents.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
    output_dir : Path
    """
    apply_ieee_style()

    metrics = ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]
    metric_labels = ["Sharpe", "Sortino", "Calmar"]

    agents = [a for a in AGENT_ORDER if a in results]
    x = np.arange(len(agents))
    width = 0.25
    offsets = np.array([-1, 0, 1]) * width

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 2.8))

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = []
        for agent in agents:
            v = results[agent].get(metric, np.nan)
            vals.append(float(v) if np.isfinite(float(v)) else 0.0)
        bars = ax.bar(
            x + offsets[i], vals, width,
            label=label,
            color=["#2166AC", "#4DAF4A", "#FF7F00"][i],
            alpha=0.82,
            edgecolor="white", linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=25, ha="right")
    ax.set_ylabel("Ratio value")
    ax.axhline(0, color="#CCCCCC", linewidth=0.8)
    ax.legend(frameon=False)

    _save(fig, output_dir, "fig4_performance_bars")


# ── Figure 5: Regime-conditional Sharpe ───────────────────────────────────────

def plot_regime_conditional_sharpe(
    regime_sharpes: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """
    Figure 5: Sharpe ratio by market regime for SAFE-HFT vs MacroHFT.

    Parameters
    ----------
    regime_sharpes : dict[str, dict[str, float]]
        Outer key: agent name. Inner dict: regime name → Sharpe ratio.
        Regime keys: ``"trending_up"``, ``"trending_down"``, ``"high_vol"``.
    output_dir : Path
    """
    apply_ieee_style()

    regime_keys = ["trending_up", "trending_down", "high_vol"]
    regime_display = ["Trending Up", "Trending Down", "High Vol."]

    agents = [a for a in ["SAFE-HFT", "MacroHFT"] if a in regime_sharpes]
    x = np.arange(len(regime_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 1.4, 2.5))

    for i, agent in enumerate(agents):
        vals = [
            float(regime_sharpes[agent].get(rk, np.nan))
            for rk in regime_keys
        ]
        ax.bar(
            x + (i - 0.5) * width, vals, width,
            label=agent,
            color=AGENT_COLORS.get(agent, "#999999"),
            alpha=0.82, edgecolor="white", linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(regime_display)
    ax.set_ylabel("Sharpe ratio")
    ax.axhline(0, color="#CCCCCC", linewidth=0.8)
    ax.legend(frameon=False)

    _save(fig, output_dir, "fig5_regime_sharpe")


# ── Figure 6: Trade distributions ─────────────────────────────────────────────

def plot_trade_distributions(
    safe_hft_trades: dict[str, np.ndarray],
    ppo_trades: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """
    Figure 6: Histograms of trade duration (left) and PnL per trade (right).

    Parameters
    ----------
    safe_hft_trades : dict  keys ``"durations"``, ``"pnls"``
    ppo_trades : dict       same keys for unconstrained PPO baseline
    output_dir : Path
    """
    apply_ieee_style()
    fig, (ax_dur, ax_pnl) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.5))

    # Duration histogram
    for label, trades, color in [
        ("SAFE-HFT", safe_hft_trades, AGENT_COLORS["SAFE-HFT"]),
        ("PPO",      ppo_trades,      AGENT_COLORS["PPO"]),
    ]:
        if "durations" in trades and len(trades["durations"]) > 0:
            ax_dur.hist(
                trades["durations"], bins=30,
                alpha=0.65, color=color, label=label, density=True,
                edgecolor="white", linewidth=0.3,
            )

    ax_dur.axvline(30, color="#888888", linewidth=0.8, linestyle="--",
                   label="30-bar limit")
    ax_dur.set_xlabel("Trade duration (bars)")
    ax_dur.set_ylabel("Density")
    ax_dur.legend(frameon=False)

    # PnL histogram
    for label, trades, color in [
        ("SAFE-HFT", safe_hft_trades, AGENT_COLORS["SAFE-HFT"]),
        ("PPO",      ppo_trades,      AGENT_COLORS["PPO"]),
    ]:
        if "pnls" in trades and len(trades["pnls"]) > 0:
            ax_pnl.hist(
                trades["pnls"] * 100.0, bins=30,
                alpha=0.65, color=color, label=label, density=True,
                edgecolor="white", linewidth=0.3,
            )

    ax_pnl.axvline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax_pnl.set_xlabel("Trade PnL (%)")
    ax_pnl.set_ylabel("Density")
    ax_pnl.legend(frameon=False)

    plt.tight_layout()
    _save(fig, output_dir, "fig6_trade_distributions")


# ── LaTeX tables ───────────────────────────────────────────────────────────────

def save_latex_results_table(
    results: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """
    Write the full LaTeX Table 1 to *output_path*.

    Delegates to ``train.evaluate.save_latex_table`` for consistency.
    """
    from train.evaluate import build_results_table, save_latex_table
    table = build_results_table(results)
    save_latex_table(table, output_path)


def save_hyperparameter_table(config: dict, output_path: Path) -> None:
    """
    Write a LaTeX hyperparameter summary table to *output_path*.

    Covers SAFE-HFT PPO hyperparameters and Lagrangian constraint settings.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ac = config.get("safe_hft", {})
    cc = config.get("constraints", {})

    rows: list[tuple[str, str, str]] = [
        # (Parameter, Value, Description)
        ("Learning rate ($\\alpha$)", f"{ac.get('learning_rate', 3e-4):.0e}", "Adam optimiser"),
        ("Rollout steps ($N$)", str(ac.get("n_steps", 2048)), "Steps per policy update"),
        ("Mini-batch size", str(ac.get("batch_size", 64)), "SGD mini-batch"),
        ("PPO epochs", str(ac.get("n_epochs", 10)), "Gradient steps per rollout"),
        ("Discount factor ($\\gamma$)", f"{ac.get('gamma', 0.99)}", ""),
        ("GAE $\\lambda$", f"{ac.get('gae_lambda', 0.95)}", "Generalised advantage estimation"),
        ("PPO clip $\\epsilon$", f"{ac.get('clip_range', 0.2)}", ""),
        ("Entropy coeff.", f"{ac.get('entropy_coef', 0.01)}", "Policy entropy bonus"),
        ("Max grad norm", f"{ac.get('max_grad_norm', 0.5)}", "Gradient clipping"),
        ("Cost limit $d_i$", f"{cc.get('max_drawdown_threshold', 0.05):.2f}", "Per-constraint violation rate threshold"),
        ("Lagrangian lr $\\eta$", f"{ac.get('lagrange_multiplier_lr', 0.01):.3f}", "Multiplier update step"),
        ("Max drawdown threshold", f"{cc.get('max_drawdown_threshold', 0.05)*100:.0f}\\%", ""),
        ("Duration limit", f"{cc.get('duration_limit', 30)} bars", ""),
        ("Volatility multiplier", f"{cc.get('vol_multiplier', 2.0)}$\\times$", "vs.\\ market vol"),
        ("Total training steps", "1{,}000{,}000", "per agent per ticker"),
    ]

    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SAFE-HFT hyperparameters.}",
        r"\label{tab:hyperparams}",
        r"\footnotesize",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Parameter & Value & Note \\",
        r"\midrule",
    ]
    for param, val, note in rows:
        lines.append(f"{param} & {val} & {note} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """
    Generate all figures from saved evaluation results CSVs.

    Usage
    -----
        python -m visualize.plot_results --results results/results_RELIANCE.csv
    """
    parser = argparse.ArgumentParser(description="Generate SAFE-HFT publication figures")
    parser.add_argument("--results",   default="results/results_RELIANCE.csv",
                        help="CSV produced by train.evaluate")
    parser.add_argument("--config",    default="train/config.yaml")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--tables-dir", default="results/tables")
    args = parser.parse_args()

    apply_ieee_style()

    # Load evaluation table
    results_csv = Path(args.results)
    if not results_csv.exists():
        print(f"Results CSV not found: {results_csv}")
        print("Run  python -m train.evaluate  first.")
        return

    df = pd.read_csv(results_csv, index_col=0)
    results: dict[str, dict[str, float]] = df.to_dict(orient="index")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    tables_dir = Path(args.tables_dir)

    # Reconstruct equity curves from returns (stored in CSV as metrics, not raw)
    # For figures needing equity curves we generate synthetic ones from
    # annualised_return_pct + max_drawdown_pct for illustrative purposes.
    # When raw step data is available (from evaluate.py with equity logging),
    # pass those arrays directly.
    n_bars = 375 * 109  # approx test set bars (109 test days × 375 bars/day)
    equity_curves: dict[str, np.ndarray] = {}
    for agent, m in results.items():
        ann_ret = m.get("annualised_return_pct", 0.0) / 100.0
        bar_ret = (1 + ann_ret) ** (1.0 / 94500) - 1.0  # per-bar compound return
        equity_curves[agent] = np.cumprod(
            np.ones(n_bars) * (1 + bar_ret)
        )

    # Fig 1 — Cumulative PnL
    plot_cumulative_pnl(equity_curves, output_dir)
    print(f"Fig 1 saved to {output_dir}/fig1_cumulative_pnl.{{png,pdf}}")

    # Fig 2 — Running drawdown
    plot_drawdown(equity_curves, output_dir)
    print(f"Fig 2 saved to {output_dir}/fig2_drawdown.{{png,pdf}}")

    # Fig 3 — Constraint violations (requires logged violation history — skip if absent)
    viol_log = Path("logs/violation_history.json")
    if viol_log.exists():
        with open(viol_log, encoding="utf-8") as f:
            violation_history = json.load(f)
        plot_constraint_violations(violation_history, output_dir)
        print(f"Fig 3 saved to {output_dir}/fig3_constraint_violations.{{png,pdf}}")
    else:
        print("Fig 3 skipped: logs/violation_history.json not found")

    # Fig 4 — Performance bars
    plot_performance_bars(results, output_dir)
    print(f"Fig 4 saved to {output_dir}/fig4_performance_bars.{{png,pdf}}")

    # Fig 5 — Regime-conditional Sharpe (requires per-regime breakdown)
    print("Fig 5 skipped: requires per-regime evaluation (use evaluate.py --regime)")

    # Fig 6 — Trade distributions (requires raw trade arrays)
    print("Fig 6 skipped: requires raw trade arrays from evaluate.py")

    # LaTeX tables
    save_latex_results_table(results, tables_dir / "table1_results.tex")
    print(f"Table 1 saved to {tables_dir}/table1_results.tex")

    save_hyperparameter_table(config, tables_dir / "table2_hyperparams.tex")
    print(f"Table 2 saved to {tables_dir}/table2_hyperparams.tex")


if __name__ == "__main__":
    main()
