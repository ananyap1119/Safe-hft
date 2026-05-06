"""
SAFE-HFT: Lagrangian-constrained PPO agent (PyTorch, no external safe-RL library).

Implements PPO + Lagrangian primal-dual update for three hard safety constraints:
  1. Maximum drawdown
  2. Position duration
  3. Realised volatility

The agent maximises cumulative reward while keeping each constraint's violation
rate below a configurable threshold d (default 5%). Risk is handled ENTIRELY by
the Lagrangian term — it is NEVER mixed into the reward signal.

Architecture
------------
Policy  : LSTM(seq=20, input=14, hidden=128) -> MLP([256,128]) -> Softmax(4)
Value   : MLP([256,256]) -> scalar   (estimates E[R])
Cost    : MLP([256,256]) -> scalar   (estimates E[C] for each constraint;
                                      separate heads per constraint)

Training algorithm: Proximal Policy Optimisation (Schulman et al. 2017) with
Lagrangian dual update (Stooke et al. 2020).

Reference
---------
Stooke et al., "Responsive Safety in Reinforcement Learning by PID Lagrangian
Constraints", ICML 2020. https://arxiv.org/abs/2007.03964

Author: [Author Name]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from constraints.lagrangian import LagrangianUpdater

logger = logging.getLogger(__name__)

N_ACTIONS: int = 4
LOOKBACK_WINDOW: int = 20
N_LOOKBACK_FEATURES: int = 14
OBS_EXTRA: int = 10          # pos_onehot(3) + pnl(1) + dd(1) + time(2) + regime(3)
OBS_SIZE: int = LOOKBACK_WINDOW * N_LOOKBACK_FEATURES + OBS_EXTRA   # 290
N_CONSTRAINTS: int = 3


# ── Neural network modules ────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    LSTM feature extractor + MLP actor for a discrete action space.

    Input observation is split into:
      - lookback_flat: first 280 elements, reshaped to (20, 14) for the LSTM.
      - extra:         remaining 10 elements (position, PnL, time, regime).

    Parameters
    ----------
    lstm_hidden : int
    mlp_hidden : list[int]
    n_actions : int
    """

    def __init__(
        self,
        lstm_hidden: int = 128,
        mlp_hidden: list[int] = None,
        n_actions: int = N_ACTIONS,
    ) -> None:
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [256, 128]
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(
            input_size=N_LOOKBACK_FEATURES,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        # MLP input: lstm_out + extra features
        mlp_in = lstm_hidden + OBS_EXTRA
        layers: list[nn.Module] = []
        prev = mlp_in
        for h in mlp_hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs : torch.Tensor  shape (batch, OBS_SIZE)

        Returns
        -------
        logits : torch.Tensor  shape (batch, n_actions)
        log_probs : torch.Tensor  shape (batch, n_actions)
        """
        lookback = obs[:, : LOOKBACK_WINDOW * N_LOOKBACK_FEATURES].reshape(
            -1, LOOKBACK_WINDOW, N_LOOKBACK_FEATURES
        )
        extra = obs[:, LOOKBACK_WINDOW * N_LOOKBACK_FEATURES :]
        lstm_out, _ = self.lstm(lookback)
        h = lstm_out[:, -1, :]          # last timestep output
        x = torch.cat([h, extra], dim=1)
        logits = self.mlp(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        return logits, log_probs

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or greedily select an action.

        Returns
        -------
        action, log_prob, entropy
        """
        _, log_probs = self.forward(obs)
        dist = Categorical(logits=log_probs)
        if deterministic:
            action = log_probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class CriticNetwork(nn.Module):
    """
    MLP critic that estimates a scalar value (reward or cost).

    Parameters
    ----------
    hidden : list[int]
    """

    def __init__(self, hidden: list[int] = None) -> None:
        super().__init__()
        if hidden is None:
            hidden = [256, 256]
        layers: list[nn.Module] = []
        prev = OBS_SIZE
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : torch.Tensor  shape (batch, OBS_SIZE)

        Returns
        -------
        torch.Tensor  shape (batch, 1)
        """
        return self.net(obs)


# ── Rollout buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Fixed-size rollout buffer for PPO with three cost signals.

    Parameters
    ----------
    n_steps : int
    gamma : float
    gae_lambda : float
    device : str
    """

    def __init__(
        self,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ) -> None:
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self._ptr = 0
        self._full = False

        self.obs = np.zeros((n_steps, OBS_SIZE), dtype=np.float32)
        self.actions = np.zeros(n_steps, dtype=np.int64)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.costs = np.zeros((n_steps, N_CONSTRAINTS), dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.cost_values = np.zeros((n_steps, N_CONSTRAINTS), dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)
        self.cost_advantages = np.zeros((n_steps, N_CONSTRAINTS), dtype=np.float32)
        self.cost_returns = np.zeros((n_steps, N_CONSTRAINTS), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        costs: np.ndarray,
        value: float,
        cost_value: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.log_probs[self._ptr] = log_prob
        self.rewards[self._ptr] = reward
        self.costs[self._ptr] = costs
        self.values[self._ptr] = value
        self.cost_values[self._ptr] = cost_value
        self.dones[self._ptr] = float(done)
        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True

    def is_full(self) -> bool:
        return self._full

    def compute_advantages(self, last_value: float, last_cost_values: np.ndarray) -> None:
        """
        Compute GAE advantages and discounted returns for reward and costs.

        Parameters
        ----------
        last_value : float
            Bootstrapped value estimate for the step after the last stored step.
        last_cost_values : np.ndarray  shape (N_CONSTRAINTS,)
        """
        last_gae = 0.0
        last_gae_c = np.zeros(N_CONSTRAINTS, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_val = last_value
                next_cval = last_cost_values
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_val = self.values[t + 1]
                next_cval = self.cost_values[t + 1]

            # Reward GAE
            delta = (
                self.rewards[t]
                + self.gamma * next_val * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

            # Cost GAE (one per constraint)
            for c in range(N_CONSTRAINTS):
                delta_c = (
                    self.costs[t, c]
                    + self.gamma * next_cval[c] * next_non_terminal
                    - self.cost_values[t, c]
                )
                last_gae_c[c] = (
                    delta_c
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_c[c]
                )
                self.cost_advantages[t, c] = last_gae_c[c]

        self.returns = self.advantages + self.values
        self.cost_returns = self.cost_advantages + self.cost_values

    def get_tensors(self, device: str) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.tensor(self.obs, dtype=torch.float32, device=device),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            "advantages": torch.tensor(self.advantages, dtype=torch.float32, device=device),
            "returns": torch.tensor(self.returns, dtype=torch.float32, device=device),
            "cost_advantages": torch.tensor(self.cost_advantages, dtype=torch.float32, device=device),
            "cost_returns": torch.tensor(self.cost_returns, dtype=torch.float32, device=device),
        }

    def reset(self) -> None:
        self._ptr = 0
        self._full = False


# ── Main agent ────────────────────────────────────────────────────────────────

class SafeHFTAgent:
    """
    Lagrangian-constrained PPO agent for NSE intraday trading.

    Parameters
    ----------
    config : dict
        Full config dict from ``config.yaml``.
    device : str
    seed : int
    """

    def __init__(
        self,
        config: dict,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.config = config
        self.device = device
        self.seed = seed

        ac = config.get("safe_hft", {})
        ag = config.get("agent", {})
        cc = config.get("constraints", {})

        self.lr = float(ac.get("learning_rate", 3e-4))
        self.n_steps = int(ac.get("n_steps", 2048))
        self.batch_size = int(ac.get("batch_size", 64))
        self.n_epochs = int(ac.get("n_epochs", 10))
        self.gamma = float(ac.get("gamma", 0.99))
        self.gae_lambda = float(ac.get("gae_lambda", 0.95))
        self.clip_range = float(ac.get("clip_range", 0.2))
        self.entropy_coef = float(ac.get("entropy_coef", 0.01))
        self.vf_coef = float(ac.get("value_loss_coef", 0.5))
        self.max_grad_norm = float(ac.get("max_grad_norm", 0.5))

        # Networks
        lstm_hidden = int(ag.get("lstm_hidden_size", 128))
        mlp_hidden = list(ag.get("mlp_hidden_sizes", [256, 128]))
        critic_hidden = list(ag.get("critic_hidden_sizes", [256, 256]))

        self.policy = PolicyNetwork(lstm_hidden, mlp_hidden).to(device)
        self.value_net = CriticNetwork(critic_hidden).to(device)
        self.cost_nets = nn.ModuleList(
            [CriticNetwork(critic_hidden) for _ in range(N_CONSTRAINTS)]
        ).to(device)

        all_params = (
            list(self.policy.parameters())
            + list(self.value_net.parameters())
            + list(self.cost_nets.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=self.lr)

        # Lagrangian multipliers
        lam_lr = float(cc.get("lagrange_multiplier_lr", 0.01))
        threshold = float(cc.get("cost_limit", 0.05))
        lam_init = float(ac.get("lambda_init", 0.0))
        self.lagrangian = LagrangianUpdater(
            n_constraints=N_CONSTRAINTS,
            threshold=threshold,
            lr=lam_lr,
            init_lambda=lam_init,
            device=device,
        )

        self.buffer = RolloutBuffer(
            n_steps=self.n_steps,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=device,
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def collect_rollout(self, env: object) -> dict[str, float]:
        """
        Collect ``n_steps`` transitions from *env* into the rollout buffer.

        Parameters
        ----------
        env : NSETradingEnv

        Returns
        -------
        dict  rollout statistics (mean_reward, mean_costs, etc.)
        """
        self.buffer.reset()
        self.policy.eval()
        self.value_net.eval()
        for net in self.cost_nets:
            net.eval()

        obs, _ = env.reset()
        episode_rewards: list[float] = []
        episode_costs: list[np.ndarray] = []
        ep_rew = 0.0
        ep_cost = np.zeros(N_CONSTRAINTS)
        ep_steps = 0

        while not self.buffer.is_full():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _ = self.policy.get_action(obs_t)
                value = self.value_net(obs_t).item()
                cost_value = np.array(
                    [net(obs_t).item() for net in self.cost_nets], dtype=np.float32
                )

            action_int = action.item()
            next_obs, reward, terminated, truncated, info = env.step(action_int)

            costs_dict = info.get("costs", {})
            cost_vec = np.array(
                [
                    costs_dict.get("cost_drawdown", 0.0),
                    costs_dict.get("cost_duration", 0.0),
                    costs_dict.get("cost_volatility", 0.0),
                ],
                dtype=np.float32,
            )

            done = terminated or truncated
            self.buffer.add(obs, action_int, log_prob.item(), reward, cost_vec, value, cost_value, done)

            ep_rew += reward
            ep_cost += cost_vec
            ep_steps += 1
            obs = next_obs

            if done:
                episode_rewards.append(ep_rew)
                # Store violation *rate* (fraction of steps), not sum — required by Lagrangian
                episode_costs.append(ep_cost / max(ep_steps, 1))
                ep_rew = 0.0
                ep_cost = np.zeros(N_CONSTRAINTS)
                ep_steps = 0
                obs, _ = env.reset()

        # Bootstrap last value
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_val = self.value_net(obs_t).item()
            last_cval = np.array([net(obs_t).item() for net in self.cost_nets], dtype=np.float32)

        self.buffer.compute_advantages(last_val, last_cval)

        mean_costs = (
            np.mean(episode_costs, axis=0)
            if episode_costs
            else np.zeros(N_CONSTRAINTS)
        )
        return {
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "mean_costs": mean_costs,
        }

    def update(self) -> dict[str, float]:
        """
        Run PPO update epochs on the current rollout buffer.

        Returns
        -------
        dict  training losses
        """
        self.policy.train()
        self.value_net.train()
        for net in self.cost_nets:
            net.train()

        data = self.buffer.get_tensors(self.device)
        adv = data["advantages"]
        # Normalise advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        losses = {"policy_loss": 0.0, "value_loss": 0.0, "cost_loss": 0.0, "entropy": 0.0}
        n_updates = 0
        n = self.n_steps

        for _ in range(self.n_epochs):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                mb = idx[start : start + self.batch_size]

                obs_mb = data["obs"][mb]
                act_mb = data["actions"][mb]
                old_lp_mb = data["log_probs"][mb]
                adv_mb = adv[mb]
                ret_mb = data["returns"][mb]
                cost_adv_mb = data["cost_advantages"][mb]
                cost_ret_mb = data["cost_returns"][mb]

                _, new_log_probs = self.policy(obs_mb)
                dist = Categorical(logits=new_log_probs)
                new_lp = dist.log_prob(act_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_mb)

                # Lagrangian penalty on cost advantages
                lambdas = self.lagrangian.lambdas  # shape (N_CONSTRAINTS,)
                # cost_adv_mb: (batch, N_CONSTRAINTS)
                cost_penalty = (lambdas.unsqueeze(0) * cost_adv_mb).sum(dim=1)  # (batch,)
                adjusted_adv = adv_mb - cost_penalty

                # PPO clip
                surr1 = ratio * adjusted_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adjusted_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                val_pred = self.value_net(obs_mb).squeeze(-1)
                value_loss = nn.functional.mse_loss(val_pred, ret_mb)

                # Cost value loss (sum over constraints)
                cost_loss = torch.tensor(0.0, device=self.device)
                for c, net in enumerate(self.cost_nets):
                    cpred = net(obs_mb).squeeze(-1)
                    cost_loss = cost_loss + nn.functional.mse_loss(cpred, cost_ret_mb[:, c])

                total_loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.vf_coef * cost_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())
                    + list(self.value_net.parameters())
                    + list(self.cost_nets.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                losses["policy_loss"] += policy_loss.item()
                losses["value_loss"] += value_loss.item()
                losses["cost_loss"] += cost_loss.item()
                losses["entropy"] += entropy.item()
                n_updates += 1

        return {k: v / max(n_updates, 1) for k, v in losses.items()}

    def train_step(self, env: object) -> dict[str, float]:
        """
        Collect one rollout, update Lagrange multipliers, run PPO update.

        Parameters
        ----------
        env : NSETradingEnv

        Returns
        -------
        dict  combined rollout + update statistics + lambda values
        """
        rollout_stats = self.collect_rollout(env)
        self.lagrangian.update(rollout_stats["mean_costs"])
        update_stats = self.update()
        return {
            **rollout_stats,
            **update_stats,
            **self.lagrangian.get_lambdas(),
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        Select an action for a single observation.

        Parameters
        ----------
        obs : np.ndarray  shape (OBS_SIZE,)
        deterministic : bool

        Returns
        -------
        int  action in {0, 1, 2, 3}
        """
        self.policy.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _, _ = self.policy.get_action(obs_t, deterministic=deterministic)
        return int(action.item())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """
        Save full agent state to *path*.

        Parameters
        ----------
        path : Path
        """
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value_net": self.value_net.state_dict(),
                "cost_nets": [net.state_dict() for net in self.cost_nets],
                "optimizer": self.optimizer.state_dict(),
                "lagrangian": self.lagrangian.state_dict(),
            },
            path,
        )

    def load(self, path: Path) -> None:
        """
        Load agent state from *path*.

        Parameters
        ----------
        path : Path
        """
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.value_net.load_state_dict(ckpt["value_net"])
        for net, sd in zip(self.cost_nets, ckpt["cost_nets"]):
            net.load_state_dict(sd)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lagrangian.load_state_dict(ckpt["lagrangian"])

    def get_lambda_values(self) -> dict[str, float]:
        """Return current Lagrange multiplier values."""
        return self.lagrangian.get_lambdas()
