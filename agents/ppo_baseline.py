"""
Standard unconstrained PPO baseline for SAFE-HFT comparison.

Same reward function and policy architecture as SAFE-HFT but NO constraints
and NO cost network. Isolates the effect of Lagrangian safety constraints.

Author: [Author Name]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

logger = logging.getLogger(__name__)


class PPOBaseline:
    """
    Thin wrapper around Stable-Baselines3 PPO.

    Parameters
    ----------
    env : gymnasium.Env
        Training environment instance.
    config : dict
        Full config dict from ``config.yaml``.
    seed : int
    """

    def __init__(self, env: object, config: dict, seed: int = 42) -> None:
        self.env = env
        self.config = config
        self.seed = seed
        self._model: PPO | None = None

    def build(self) -> None:
        """Instantiate the SB3 PPO model."""
        c = self.config.get("ppo_baseline", {})
        self._model = PPO(
            policy=c.get("policy", "MlpPolicy"),
            env=self.env,
            learning_rate=float(c.get("learning_rate", 3e-4)),
            n_steps=int(c.get("n_steps", 2048)),
            batch_size=int(c.get("batch_size", 64)),
            n_epochs=int(c.get("n_epochs", 10)),
            gamma=float(c.get("gamma", 0.99)),
            gae_lambda=float(c.get("gae_lambda", 0.95)),
            clip_range=float(c.get("clip_range", 0.2)),
            ent_coef=float(c.get("ent_coef", 0.01)),
            vf_coef=float(c.get("vf_coef", 0.5)),
            max_grad_norm=float(c.get("max_grad_norm", 0.5)),
            seed=self.seed,
            verbose=0,
        )

    def train(self, total_steps: int, checkpoint_dir: Path) -> None:
        """
        Train for *total_steps* environment steps.

        Parameters
        ----------
        total_steps : int
        checkpoint_dir : Path
        """
        if self._model is None:
            self.build()
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._model.learn(total_timesteps=total_steps, progress_bar=False)
        self._model.save(checkpoint_dir / "ppo_final")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        Returns
        -------
        int  action in {0, 1, 2, 3}
        """
        if self._model is None:
            raise RuntimeError("Call build() and train() before predict().")
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return int(action)

    def save(self, path: Path) -> None:
        self._model.save(path)

    def load(self, path: Path) -> None:
        self._model = PPO.load(path, env=self.env)
