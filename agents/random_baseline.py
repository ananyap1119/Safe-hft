"""
Uniform random action baseline for SAFE-HFT comparison.

This is the performance floor: every agent must beat it convincingly.
No learning occurs — actions are drawn uniformly from {0, 1, 2, 3} at every step.

Author: [Author Name]
"""

from __future__ import annotations

import numpy as np


class RandomBaseline:
    """
    Stateless random agent.

    Parameters
    ----------
    n_actions : int
        Size of the discrete action space. Default: 4.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_actions: int = 4, seed: int = 42) -> None:
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Sample a uniformly random action.

        Parameters
        ----------
        obs : np.ndarray
            Current observation (ignored).
        deterministic : bool
            Ignored; always samples randomly.

        Returns
        -------
        int
            Random action in ``{0, ..., n_actions - 1}``.
        """
        return int(self.rng.integers(0, self.n_actions))
