"""
Main training orchestration for SAFE-HFT.

Trains all agents on the training split of the NSE dataset. All hyperparameters
are read from train/config.yaml — nothing is hardcoded.

Training protocol
-----------------
- 1,000,000 environment steps per agent.
- Fixed seeds: numpy.random.seed(42), torch.manual_seed(42), random.seed(42).
- Training data: 70% chronological prefix.
- Validation every eval_interval steps on val split (Sharpe-based checkpoint selection).
- Best checkpoint kept per agent.
- All metrics logged to Weights & Biases (one run per agent).

Usage
-----
    python -m train.train --agent safe_hft
    python -m train.train --agent all
    python -m train.train --agent ppo --ticker TCS

Author: [Author Name]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)

AGENTS: list[str] = ["safe_hft", "ppo", "dqn", "macrohft", "earnhft", "random"]
TICKERS: list[str] = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "datasets" / "processed"


def set_global_seeds(seed: int = 42) -> None:
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> dict:
    """Load and return the YAML config dict."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = {"data", "env", "agent", "safe_hft", "training", "constraints"}
    missing = required - set(cfg.keys())
    if missing:
        raise KeyError(f"Config missing sections: {missing}")
    return cfg


def load_env_data(ticker: str, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    Load processed parquet and split indices for *ticker*.

    Returns
    -------
    df : pd.DataFrame
    splits : dict
    """
    df_path = DATA_DIR / f"{ticker}_processed.parquet"
    split_path = DATA_DIR / f"{ticker}_splits.json"
    if not df_path.exists():
        raise FileNotFoundError(f"No processed data for {ticker}. Run data/preprocess.py first.")
    df = pd.read_parquet(df_path)
    with open(split_path, encoding="utf-8") as f:
        splits = json.load(f)
    return df, splits


def build_env(ticker: str, config: dict, mode: str) -> object:
    """Construct a fully initialised NSETradingEnv."""
    from env.trading_env import NSETradingEnv
    df, splits = load_env_data(ticker, config)
    env_cfg = {}
    env_cfg.update(config.get("constraints", {}))
    env_cfg.update(config.get("env", {}))
    return NSETradingEnv(df, splits, env_cfg, ticker=ticker, mode=mode,
                         seed=config["agent"]["seed"])


def validate_agent(agent: object, val_env: object) -> dict[str, float]:
    """
    Run one deterministic episode on the validation split and return metrics.

    Parameters
    ----------
    agent : object  any agent with a predict(obs, deterministic=True) method
    val_env : NSETradingEnv

    Returns
    -------
    dict  {"sharpe_ratio": ..., "max_drawdown_pct": ...}
    """
    from metrics.finance_metrics import sharpe_ratio, max_drawdown

    returns: list[float] = []
    obs, _ = val_env.reset(seed=0)
    while True:
        action = agent.predict(obs, deterministic=True) if hasattr(agent, "predict") else val_env.action_space.sample()
        obs, rew, term, trunc, _ = val_env.step(action)
        returns.append(rew)
        if term or trunc:
            # Run through all val days
            if hasattr(val_env, "_day_cursor") and val_env._day_cursor < val_env._n_days:
                obs, _ = val_env.reset()
            else:
                break

    arr = np.array(returns, dtype=np.float64)
    return {
        "sharpe_ratio": sharpe_ratio(arr),
        "max_drawdown_pct": max_drawdown(arr) * 100.0,
    }


def train_safe_hft(config: dict, ticker: str, output_dir: Path) -> None:
    """Train the SAFE-HFT Lagrangian PPO agent."""
    from agents.safe_hft_agent import SafeHFTAgent

    train_env = build_env(ticker, config, "train")
    val_env = build_env(ticker, config, "val")

    tc = config["training"]
    total_steps = int(tc.get("total_steps", 1_000_000))
    eval_interval = int(tc.get("eval_interval", 100_000))
    ckpt_interval = int(tc.get("checkpoint_interval", 100_000))
    best_metric = tc.get("best_metric", "sharpe_ratio")

    ckpt_dir = output_dir / "safe_hft" / ticker
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = config["agent"].get("device", "cpu")
    agent = SafeHFTAgent(config, device=device, seed=config["agent"]["seed"])

    best_val_score = -np.inf
    steps_done = 0
    rollout_num = 0

    logger.info("[%s] safe_hft training started (%d total steps)", ticker, total_steps)

    try:
        import wandb
        run = wandb.init(
            project=tc.get("wandb_project", "safe-hft"),
            name=f"safe_hft_{ticker}",
            config=config,
            reinit=True,
        )
        use_wandb = True
    except Exception:
        use_wandb = False

    n_steps = agent.n_steps
    while steps_done < total_steps:
        stats = agent.train_step(train_env)
        steps_done += n_steps
        rollout_num += 1

        log = {
            "steps": steps_done,
            "mean_reward": stats["mean_reward"],
            "mean_costs": stats["mean_costs"].tolist() if hasattr(stats["mean_costs"], "tolist") else stats["mean_costs"],
            **{k: v for k, v in stats.items() if k.startswith("lambda_")},
            "policy_loss": stats.get("policy_loss", 0),
            "value_loss": stats.get("value_loss", 0),
        }

        if steps_done % ckpt_interval < n_steps:
            ckpt_path = ckpt_dir / f"step_{steps_done:08d}.pt"
            agent.save(ckpt_path)

        if steps_done % eval_interval < n_steps:
            val_stats = validate_agent(agent, val_env)
            log.update({f"val_{k}": v for k, v in val_stats.items()})
            score = val_stats.get(best_metric, -np.inf)
            if np.isfinite(score) and score > best_val_score:
                best_val_score = score
                agent.save(ckpt_dir / "best.pt")
                logger.info(
                    "[%s] safe_hft step %d  val_%s=%.4f  (NEW BEST)",
                    ticker, steps_done, best_metric, score,
                )

        if use_wandb:
            run.log(log)

        if rollout_num % 10 == 0:
            lam = agent.get_lambda_values()
            logger.info(
                "[%s] safe_hft step %7d / %d  reward=%.4f  "
                "lam_dd=%.3f lam_dur=%.3f lam_vol=%.3f",
                ticker, steps_done, total_steps,
                stats["mean_reward"],
                lam["lambda_drawdown"],
                lam["lambda_duration"],
                lam["lambda_volatility"],
            )

    if use_wandb:
        run.finish()
    logger.info("[%s] safe_hft training complete. Best val %s=%.4f", ticker, best_metric, best_val_score)


def train_sb3_agent(agent_name: str, config: dict, ticker: str, output_dir: Path) -> None:
    """Train a Stable-Baselines3 baseline (ppo or dqn)."""
    from agents.ppo_baseline import PPOBaseline
    from agents.dqn_baseline import DQNBaseline

    train_env = build_env(ticker, config, "train")
    tc = config["training"]
    total_steps = int(tc.get("total_steps", 1_000_000))
    ckpt_dir = output_dir / agent_name / ticker
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    AgentClass = PPOBaseline if agent_name == "ppo" else DQNBaseline
    agent = AgentClass(train_env, config, seed=config["agent"]["seed"])
    agent.build()
    agent.train(total_steps, ckpt_dir)
    logger.info("[%s] %s training complete.", ticker, agent_name)


def train_macrohft(config: dict, ticker: str, output_dir: Path) -> None:
    """Train the simplified MacroHFT-style baseline."""
    from agents.macrohft_baseline import MacroHFTBaseline

    # Both sub-envs use the same full train environment in this simplified version
    train_env = build_env(ticker, config, "train")
    tc = config["training"]
    total_steps = int(tc.get("total_steps", 1_000_000))
    ckpt_dir = output_dir / "macrohft" / ticker
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    agent = MacroHFTBaseline(
        env_trending=train_env,
        env_volatile=train_env,
        env_full=train_env,
        config=config,
        seed=config["agent"]["seed"],
    )
    agent.train(total_steps, ckpt_dir)
    logger.info("[%s] macrohft training complete.", ticker)


def train_earnhft(config: dict, ticker: str, output_dir: Path) -> None:
    """Train the simplified EarnHFT-style baseline."""
    from agents.earnhft_baseline import EarnHFTBaseline

    train_env = build_env(ticker, config, "train")
    tc = config["training"]
    total_steps = int(tc.get("total_steps", 1_000_000))
    ckpt_dir = output_dir / "earnhft" / ticker
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    agent = EarnHFTBaseline(train_env, config, seed=config["agent"]["seed"])
    agent.build()
    agent.train(total_steps, ckpt_dir)
    logger.info("[%s] earnhft training complete.", ticker)


def train_agent(agent_name: str, config: dict, ticker: str, output_dir: Path) -> None:
    """Dispatch to the correct training function."""
    dispatch = {
        "safe_hft": train_safe_hft,
        "ppo": lambda c, t, o: train_sb3_agent("ppo", c, t, o),
        "dqn": lambda c, t, o: train_sb3_agent("dqn", c, t, o),
        "macrohft": train_macrohft,
        "earnhft": train_earnhft,
        "random": lambda c, t, o: logger.info("Random baseline needs no training."),
    }
    if agent_name not in dispatch:
        raise ValueError(f"Unknown agent: {agent_name}. Choose from {AGENTS}")
    dispatch[agent_name](config, ticker, output_dir)


def main() -> None:
    """Entry point: parse CLI args, set seeds, dispatch training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train SAFE-HFT agents")
    parser.add_argument("--config", default="train/config.yaml")
    parser.add_argument("--agent", default="safe_hft",
                        choices=AGENTS + ["all"],
                        help="Agent to train ('all' trains all agents sequentially)")
    parser.add_argument("--ticker", default="RELIANCE",
                        choices=TICKERS + ["all"],
                        help="Ticker to train on ('all' trains on all 5 tickers)")
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    set_global_seeds(config["agent"]["seed"])

    output_dir = ROOT / args.output_dir
    agents_to_train = AGENTS if args.agent == "all" else [args.agent]
    tickers_to_train = TICKERS if args.ticker == "all" else [args.ticker]

    for ticker in tickers_to_train:
        for agent_name in agents_to_train:
            logger.info("=" * 60)
            logger.info("Training  agent=%-10s  ticker=%s", agent_name, ticker)
            logger.info("=" * 60)
            try:
                train_agent(agent_name, config, ticker, output_dir)
            except Exception as exc:
                logger.error("FAILED: %s / %s — %s", agent_name, ticker, exc, exc_info=True)


if __name__ == "__main__":
    main()
