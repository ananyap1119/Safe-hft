"""
Lagrange multiplier management for SAFE-HFT.

Implements the primal-dual (Lagrangian) constraint update rule:

    lambda_i <- max(0,  lambda_i + alpha * (mean_episode_cost_i - d_i))

where:
  alpha  = multiplier learning rate  (default 0.01)
  d_i    = per-constraint violation-rate threshold  (default 0.05)

Reference: Stooke et al., "Responsive Safety in Reinforcement Learning by PID
Lagrangian Constraints", ICML 2020.

Author: [Author Name]
"""

from __future__ import annotations

import numpy as np
import torch


CONSTRAINT_NAMES: list[str] = [
    "lambda_drawdown",
    "lambda_duration",
    "lambda_volatility",
]


class LagrangianUpdater:
    """
    Lagrange multiplier manager for three safety constraints.

    Used both inside ``LagrangianPPO`` during training and standalone for
    unit tests and offline analysis.

    Parameters
    ----------
    n_constraints : int
        Number of safety constraints (default: 3).
    threshold : float
        Target maximum violation rate per constraint (default: 0.05).
    lr : float
        Multiplier update learning rate alpha (default: 0.01).
    init_lambda : float
        Initial value for all multipliers (default: 0.0).
    device : str
        Torch device for tensor operations.
    """

    def __init__(
        self,
        n_constraints: int = 3,
        threshold: float = 0.05,
        lr: float = 0.01,
        init_lambda: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.n_constraints = n_constraints
        self.threshold = threshold
        self.lr = lr
        self.device = device
        self.lambdas: torch.Tensor = torch.full(
            (n_constraints,), init_lambda, dtype=torch.float32, device=device
        )

    def update(self, mean_episode_costs: np.ndarray) -> np.ndarray:
        """
        Perform one Lagrangian multiplier update step.

        lambda_i <- max(0, lambda_i + lr * (cost_i - threshold))

        Parameters
        ----------
        mean_episode_costs : np.ndarray
            Shape ``(n_constraints,)``. Mean constraint violation rate for each
            constraint over the last rollout batch.

        Returns
        -------
        np.ndarray
            Updated multiplier values, shape ``(n_constraints,)``.
        """
        costs = torch.tensor(mean_episode_costs, dtype=torch.float32, device=self.device)
        self.lambdas = torch.clamp(
            self.lambdas + self.lr * (costs - self.threshold), min=0.0
        )
        return self.lambdas.cpu().numpy()

    def get_lambdas(self) -> dict[str, float]:
        """
        Return named multiplier values.

        Returns
        -------
        dict[str, float]
        """
        vals = self.lambdas.cpu().numpy()
        return {name: float(vals[i]) for i, name in enumerate(CONSTRAINT_NAMES)}

    def penalty(self) -> torch.Tensor:
        """
        Return total Lagrangian penalty weight (sum of lambdas) as a tensor.

        Used to scale the cost advantage during the PPO update.

        Returns
        -------
        torch.Tensor  scalar
        """
        return self.lambdas.sum()

    def reset(self) -> None:
        """Reset all multipliers to their initial value."""
        self.lambdas.zero_()

    def state_dict(self) -> dict:
        return {"lambdas": self.lambdas.cpu().numpy().tolist()}

    def load_state_dict(self, d: dict) -> None:
        self.lambdas = torch.tensor(d["lambdas"], dtype=torch.float32, device=self.device)
