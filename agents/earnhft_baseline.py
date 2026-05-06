"""
Simplified EarnHFT-style hierarchical baseline for SAFE-HFT comparison.

Inspired by:
  Qin et al., "EarnHFT: Efficient Hierarchical Reinforcement Learning for
  High Frequency Trading", AAAI 2024.

Architecture (simplified re-implementation)
--------------------------------------------
- High-level agent (PPO): decides a directional signal every ``high_level_interval``
  bars — one of {BUY, SELL, HOLD_SIGNAL}. Trained with a slower learning rate.
- Low-level agent (PPO): executes fine-grained actions every bar, conditioned on
  the high-level signal. The signal is appended to the observation as a 3-element
  one-hot vector, extending obs from 290 → 293 dims.

Simplifications vs the original paper
--------------------------------------
- Original EarnHFT targets crypto; we adapt to NSE equities.
- We omit the original paper's option pricing reward shaping.
- We use SB3 PPO for both levels rather than the paper's custom Q-learning.

Cite: Qin et al., AAAI 2024.

Author: [Author Name]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

N_HL_SIGNALS: int = 3   # BUY / SELL / HOLD_SIGNAL
HL_SIGNAL_BUY: int = 0
HL_SIGNAL_SELL: int = 1
HL_SIGNAL_HOLD: int = 2


class _LowLevelWrapper(gym.Env):
    """
    Wraps the base environment to append the high-level signal to every
    observation, extending obs_size by N_HL_SIGNALS.
    """

    metadata: dict = {"render_modes": []}

    def __init__(self, env: object, hl_agent: "PPO | None") -> None:
        self._env = env
        self._hl_agent = hl_agent
        self._current_signal: int = HL_SIGNAL_HOLD
        self._step_in_interval: int = 0

        base_obs = env.observation_space
        new_size = base_obs.shape[0] + N_HL_SIGNALS
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_size,), dtype=np.float32
        )
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        self._current_signal = HL_SIGNAL_HOLD
        self._step_in_interval = 0
        return self._augment(obs), info

    def step(self, action):
        obs, rew, term, trunc, info = self._env.step(action)
        self._step_in_interval += 1
        if self._step_in_interval >= self._env.config.get("high_level_interval", 10):
            self._step_in_interval = 0
            if self._hl_agent is not None:
                aug_obs = self._augment(obs)
                hl_action, _ = self._hl_agent.predict(aug_obs, deterministic=False)
                self._current_signal = int(hl_action)
        return self._augment(obs), rew, term, trunc, info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        sig_onehot = np.zeros(N_HL_SIGNALS, dtype=np.float32)
        sig_onehot[self._current_signal] = 1.0
        return np.concatenate([obs, sig_onehot])

    def render(self): pass
    def close(self): pass


class EarnHFTBaseline:
    """
    Simplified two-level hierarchical RL baseline inspired by EarnHFT.

    Parameters
    ----------
    env : gymnasium.Env
        Base trading environment.
    config : dict
    seed : int
    """

    def __init__(self, env: object, config: dict, seed: int = 42) -> None:
        self.env = env
        self.config = config
        self.seed = seed
        self._hl_agent: PPO | None = None
        self._ll_agent: PPO | None = None
        self._ll_env: _LowLevelWrapper | None = None

    def build(self) -> None:
        """Instantiate both high-level and low-level PPO agents."""
        c = self.config.get("earnhft_baseline", {})
        gamma = float(c.get("gamma", 0.99))

        # High-level: 3-action space (BUY / SELL / HOLD_SIGNAL)
        hl_env = _HighLevelEnvWrapper(self.env)
        self._hl_agent = PPO(
            "MlpPolicy", hl_env,
            learning_rate=float(c.get("high_level_lr", 1e-4)),
            gamma=gamma, seed=self.seed, verbose=0,
        )

        # Low-level: base env + signal one-hot
        self._ll_env = _LowLevelWrapper(self.env, self._hl_agent)
        self._ll_agent = PPO(
            "MlpPolicy", self._ll_env,
            learning_rate=float(c.get("low_level_lr", 3e-4)),
            gamma=gamma, seed=self.seed, verbose=0,
        )

    def train(self, total_steps: int, checkpoint_dir: Path) -> None:
        """
        Train HL agent first (20% budget), then LL agent (80% budget).

        Parameters
        ----------
        total_steps : int
        checkpoint_dir : Path
        """
        if self._hl_agent is None:
            self.build()
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        hl_steps = total_steps // 5
        ll_steps = total_steps - hl_steps

        logger.info("EarnHFT: training high-level agent for %d steps", hl_steps)
        self._hl_agent.learn(total_timesteps=hl_steps, progress_bar=False)
        self._hl_agent.save(checkpoint_dir / "earnhft_hl")

        logger.info("EarnHFT: training low-level agent for %d steps", ll_steps)
        self._ll_agent.learn(total_timesteps=ll_steps, progress_bar=False)
        self._ll_agent.save(checkpoint_dir / "earnhft_ll")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        Parameters
        ----------
        obs : np.ndarray  shape (290,)  base obs
        deterministic : bool

        Returns
        -------
        int  action in {0, 1, 2, 3}
        """
        if self._ll_agent is None:
            raise RuntimeError("Call train() before predict().")
        sig_onehot = np.zeros(N_HL_SIGNALS, dtype=np.float32)
        sig_onehot[self._ll_env._current_signal] = 1.0
        aug_obs = np.concatenate([obs, sig_onehot])
        action, _ = self._ll_agent.predict(aug_obs, deterministic=deterministic)
        return int(action)

    def save(self, path: Path) -> None:
        path = Path(path)
        self._hl_agent.save(str(path) + "_hl")
        self._ll_agent.save(str(path) + "_ll")

    def load(self, path: Path) -> None:
        path = Path(path)
        self._ll_env = _LowLevelWrapper(self.env, None)
        self._hl_agent = PPO.load(str(path) + "_hl")
        self._ll_env._hl_agent = self._hl_agent
        self._ll_agent = PPO.load(str(path) + "_ll", env=self._ll_env)


class _HighLevelEnvWrapper(gym.Env):
    """
    Wraps the base env to present a 3-action space (BUY / SELL / HOLD) to the
    high-level agent, aggregating reward over ``high_level_interval`` steps.
    """

    metadata: dict = {"render_modes": []}

    def __init__(self, env: object) -> None:
        self._env = env
        self._interval = env.config.get("high_level_interval", 10)
        self.observation_space = env.observation_space
        self.action_space = spaces.Discrete(N_HL_SIGNALS)

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def step(self, hl_action: int):
        # Map HL signal to LL action and step interval times
        ll_action = {HL_SIGNAL_BUY: 1, HL_SIGNAL_SELL: 2, HL_SIGNAL_HOLD: 0}[hl_action]
        total_rew = 0.0
        obs, info = None, {}
        for _ in range(self._interval):
            obs, rew, term, trunc, info = self._env.step(ll_action)
            total_rew += rew
            ll_action = 0   # hold for subsequent steps within interval
            if term or trunc:
                return obs, total_rew, term, trunc, info
        return obs, total_rew, False, False, info

    def render(self): pass
    def close(self): pass
