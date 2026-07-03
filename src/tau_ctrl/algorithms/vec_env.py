"""GPU-vectorized environments — the piece that unlocks real GPU speedup.

Single-environment training loops (and tau-ctrl's single-env ``learn()``) step
one CPU environment at a time: one transition, one tiny update, ~100 sequential
CUDA kernel launches per step.  At that granularity kernel-launch latency dwarfs
the actual math, so "putting the network on the GPU" caps out at ~1.3x.

A :class:`TorchVecEnv` instead advances ``num_envs`` environments *in
parallel*, entirely as batched tensor ops on the target device.  Nothing
crosses the Python/numpy boundary in the hot loop, and each step yields
``num_envs`` transitions that feed a large batched update — which is what
actually saturates a GPU (10-50x once ``num_envs`` is in the hundreds).

The interface deliberately mirrors gymnasium's, but batched and tensor-native:

    env = TorchPendulum(num_envs=1024, device="cuda")
    obs = env.reset()                       # (N, obs_dim) tensor on device
    next_obs, reward, done, start_obs = env.step(action)   # all (N, ...) tensors

``step`` returns the *true* transition target (``next_obs``, ``done`` = the
terminal observation / termination flag used for bootstrapping) plus
``start_obs`` — the observation to continue from, with any done envs already
auto-reset.  Off-policy trainers store ``(obs, act, reward, next_obs, done)``
and carry ``start_obs`` forward; see ``off_policy.ReplayBuffer.add_batch``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError("tau_ctrl vectorized envs require gymnasium") from e


class TorchVecEnv:
    """Base class for on-device, batched environments.

    Subclasses implement the batched dynamics in :meth:`_reset_idx` and
    :meth:`step`.  ``observation_space``/``action_space`` describe a *single*
    env (so torch controllers read dims/bounds exactly as for a gym env); the
    batch dimension is ``num_envs``.
    """

    observation_space: spaces.Box
    action_space: spaces.Box

    def __init__(self, num_envs: int, device: str = "cpu") -> None:
        import torch  # noqa: PLC0415

        self._torch = torch
        self.num_envs = int(num_envs)
        self.device = device

    # -- subclass hooks ---------------------------------------------------
    def _obs(self):  # -> (N, obs_dim) tensor
        raise NotImplementedError

    def _reset_idx(self, mask) -> None:
        """Reset the envs selected by boolean ``mask`` (shape (N,))."""
        raise NotImplementedError

    # -- gym-like API (batched) ------------------------------------------
    def reset(self):
        """Reset every env; return obs tensor ``(N, obs_dim)`` on device."""
        all_ = self._torch.ones(self.num_envs, dtype=self._torch.bool, device=self.device)
        self._reset_idx(all_)
        return self._obs()

    def step(self, actions):
        """Advance all envs one step.

        Returns ``(next_obs, reward, done, start_obs)`` — all device tensors:
          * ``next_obs`` (N, obs_dim): true post-dynamics observation.
          * ``reward``   (N,): per-env reward.
          * ``done``     (N,) float: termination flag for bootstrapping
            (time-limit truncation is *not* a done here — it doesn't break the
            value bootstrap — but still triggers an auto-reset).
          * ``start_obs`` (N, obs_dim): observation to continue from, with any
            terminated/truncated envs already reset.
        """
        raise NotImplementedError


class TorchPendulum(TorchVecEnv):
    """Vectorized torque-limited pendulum — the batched twin of
    :class:`~tau_ctrl.algorithms.envs.toy.PendulumSwingUp`.

    Identical dynamics/reward (angle 0 = upright), run for ``num_envs`` envs at
    once as pure tensor ops.  Fixed-horizon episodes (``max_episode_steps``)
    auto-reset; there is no true termination, so ``done`` is always 0 (correct
    for bootstrapping a time-limit truncation).
    """

    def __init__(
        self,
        num_envs: int = 1024,
        device: str = "cpu",
        max_torque: float = 6.0,
        dt: float = 0.05,
        g: float = 10.0,
        m: float = 1.0,
        l: float = 1.0,
        start_angle: float = np.pi,
        max_speed: float = 8.0,
        max_episode_steps: int = 200,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(num_envs, device)
        t = self._torch
        self.max_torque = float(max_torque)
        self.max_speed = float(max_speed)
        self.dt, self.g, self.m, self.l = dt, g, m, l
        self.start_angle = float(start_angle)
        self.max_episode_steps = int(max_episode_steps)

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            -self.max_torque, self.max_torque, (1,), np.float32
        )

        self._gen = t.Generator(device=device)
        if seed is not None:
            self._gen.manual_seed(int(seed))
        # State: theta (from upright), theta_dot, elapsed steps — all on device.
        self.th = t.zeros(self.num_envs, device=device)
        self.thd = t.zeros(self.num_envs, device=device)
        self.step_count = t.zeros(self.num_envs, dtype=t.long, device=device)

    def _obs(self):
        t = self._torch
        return t.stack([t.cos(self.th), t.sin(self.th), self.thd], dim=-1)

    def _reset_idx(self, mask) -> None:
        t = self._torch
        n = int(mask.sum().item())
        if n == 0:
            return
        # Small random perturbation around the hanging-down start, like the
        # scalar env's fixed start but batched (keeps envs decorrelated).
        noise = (t.rand(n, generator=self._gen, device=self.device) - 0.5) * 0.1
        self.th[mask] = self.start_angle + noise
        self.thd[mask] = 0.0
        self.step_count[mask] = 0

    def step(self, actions):
        t = self._torch
        u = actions.reshape(self.num_envs).clamp(-self.max_torque, self.max_torque)
        thddot = (self.g / self.l) * t.sin(self.th) + u / (self.m * self.l**2)
        self.thd = (self.thd + thddot * self.dt).clamp(-self.max_speed, self.max_speed)
        self.th = self.th + self.thd * self.dt
        self.th = ((self.th + np.pi) % (2 * np.pi)) - np.pi  # wrap to [-pi, pi]

        cost = self.th**2 + 0.1 * self.thd**2 + 1e-3 * u**2
        reward = -cost
        next_obs = self._obs()

        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        done = t.zeros(self.num_envs, device=self.device)  # no true termination

        # Auto-reset truncated envs; start_obs is what the trainer continues from.
        start_obs = next_obs
        if bool(truncated.any()):
            self._reset_idx(truncated)
            start_obs = next_obs.clone()
            start_obs[truncated] = self._obs()[truncated]
        return next_obs, reward, done, start_obs
