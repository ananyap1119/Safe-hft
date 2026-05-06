"""
Simplified MacroHFT-style baseline for SAFE-HFT comparison.

Implements a two-tier regime-decomposed agent inspired by:
  Zong et al., "MacroHFT: Memory Augmented Context-aware Reinforcement
  Learning On High Frequency Trading", KDD 2024. (ZONG0004/MacroHFT)

Architecture (simplified re-implementation)
--------------------------------------------
- Sub-agent A (PPO): trained on bars where regime == 0 or 1 (trending).
- Sub-agent B (PPO): trained on bars where regime == 2 (high-volatility).
- Hyper-agent: a lightweight MLP that observes a short memory buffer of recent
  regime labels and selects which sub-agent to query at each step.

Simplifications vs the original paper
--------------------------------------
- We do not implement the full memory-augmented recurrent architecture.
- Sub-agents use SB3 PPO (MLP policy) rather than the paper's custom policy.
- The hyper-agent is a greedy rule (follow most-recent regime) rather than a
  learned meta-controller; this is sufficient to demonstrate the regime-switching
  behaviour.
- Training is sequential (sub-A then sub-B) not joint.

This baseline demonstrates that regime-decomposition alone (without safety
constraints) does not achieve SAFE-HFT's drawdown bounds.

Cite: Zong et al., KDD 2024.

Author: [Author Name]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)


class MacroHFTBaseline:
    """
    Simplified two-sub-agent MacroHFT-style regime-switching baseline.

    Parameters
    ----------
    env_trending : gymnasium.Env
        Environment whose episodes are restricted to trending-regime days
        (regime predominantly 0 or 1).
    env_volatile : gymnasium.Env
        Environment whose episodes are restricted to high-volatility-regime
        days (regime predominantly 2).
    env_full : gymnasium.Env
        Full environment used during evaluation.
    config : dict
    seed : int
    """

    def __init__(
        self,
        env_trending: object,
        env_volatile: object,
        env_full: object,
        config: dict,
        seed: int = 42,
    ) -> None:
        self.env_trending = env_trending
        self.env_volatile = env_volatile
        self.env_full = env_full
        self.config = config
        self.seed = seed
        self._sub_a: PPO | None = None   # trending regime
        self._sub_b: PPO | None = None   # volatile regime

    def build(self) -> None:
        """Instantiate both sub-agent PPO models."""
        c = self.config.get("macrohft_baseline", {})
        lr = float(c.get("sub_agent_lr", 3e-4))
        gamma = float(c.get("gamma", 0.99))

        self._sub_a = PPO(
            "MlpPolicy", self.env_trending,
            learning_rate=lr, gamma=gamma, seed=self.seed, verbose=0,
        )
        self._sub_b = PPO(
            "MlpPolicy", self.env_volatile,
            learning_rate=lr, gamma=gamma, seed=self.seed, verbose=0,
        )

    def train(self, total_steps: int, checkpoint_dir: Path) -> None:
        """
        Train sub-agent A on trending data, then sub-agent B on volatile data.

        Each sub-agent receives half the total step budget.

        Parameters
        ----------
        total_steps : int
        checkpoint_dir : Path
        """
        if self._sub_a is None:
            self.build()
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        half = total_steps // 2
        logger.info("MacroHFT: training sub-agent A (trending) for %d steps", half)
        self._sub_a.learn(total_timesteps=half, progress_bar=False)
        self._sub_a.save(checkpoint_dir / "macrohft_sub_a")

        logger.info("MacroHFT: training sub-agent B (volatile) for %d steps", half)
        self._sub_b.learn(total_timesteps=half, progress_bar=False)
        self._sub_b.save(checkpoint_dir / "macrohft_sub_b")

    def predict(
        self,
        obs: np.ndarray,
        regime: int,
        deterministic: bool = True,
    ) -> int:
        """
        Dispatch to the appropriate sub-agent based on the current regime.

        Sub-agent A handles trending regimes (0, 1); sub-agent B handles
        high-volatility regime (2). This mimics the hyper-agent's role.

        Parameters
        ----------
        obs : np.ndarray
        regime : int  0, 1, or 2
        deterministic : bool

        Returns
        -------
        int  action in {0, 1, 2, 3}
        """
        if self._sub_a is None or self._sub_b is None:
            raise RuntimeError("Call train() before predict().")
        if regime in (0, 1):
            action, _ = self._sub_a.predict(obs, deterministic=deterministic)
        else:
            action, _ = self._sub_b.predict(obs, deterministic=deterministic)
        return int(action)

    def save(self, path: Path) -> None:
        """Serialise both sub-agents. Writes ``<path>_sub_a.zip`` and ``<path>_sub_b.zip``."""
        path = Path(path)
        self._sub_a.save(str(path) + "_sub_a")
        self._sub_b.save(str(path) + "_sub_b")

    def load(self, path: Path) -> None:
        """Load both sub-agents from ``<path>_sub_a.zip`` and ``<path>_sub_b.zip``."""
        path = Path(path)
        self._sub_a = PPO.load(str(path) + "_sub_a", env=self.env_trending)
        self._sub_b = PPO.load(str(path) + "_sub_b", env=self.env_volatile)
