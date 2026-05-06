"""
DQN baseline agent for SAFE-HFT comparison.

Uses Stable-Baselines3's DQN with an MLP policy. No constraints.

Author: [Author Name]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

logger = logging.getLogger(__name__)


class DQNBaseline:
    """
    Thin wrapper around SB3 DQN.

    Parameters
    ----------
    env : gymnasium.Env
    config : dict
    seed : int
    """

    def __init__(self, env: object, config: dict, seed: int = 42) -> None:
        self.env = env
        self.config = config
        self.seed = seed
        self._model: DQN | None = None

    def build(self) -> None:
        """Instantiate the SB3 DQN model."""
        c = self.config.get("dqn_baseline", {})
        self._model = DQN(
            policy=c.get("policy", "MlpPolicy"),
            env=self.env,
            learning_rate=float(c.get("learning_rate", 1e-4)),
            batch_size=int(c.get("batch_size", 32)),
            buffer_size=int(c.get("buffer_size", 50000)),
            learning_starts=int(c.get("learning_starts", 1000)),
            gamma=float(c.get("gamma", 0.99)),
            tau=float(c.get("tau", 1.0)),
            train_freq=int(c.get("train_freq", 4)),
            gradient_steps=int(c.get("gradient_steps", 1)),
            target_update_interval=int(c.get("target_update_interval", 1000)),
            exploration_fraction=float(c.get("exploration_fraction", 0.1)),
            exploration_final_eps=float(c.get("exploration_final_eps", 0.05)),
            seed=self.seed,
            verbose=0,
        )

    def train(self, total_steps: int, checkpoint_dir: Path) -> None:
        if self._model is None:
            self.build()
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self._model.learn(total_timesteps=total_steps, progress_bar=False)
        self._model.save(Path(checkpoint_dir) / "dqn_final")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        if self._model is None:
            raise RuntimeError("Call build() and train() before predict().")
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return int(action)

    def save(self, path: Path) -> None:
        self._model.save(path)

    def load(self, path: Path) -> None:
        self._model = DQN.load(path, env=self.env)
