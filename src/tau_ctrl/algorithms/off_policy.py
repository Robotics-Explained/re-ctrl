"""Shared machinery for off-policy actor-critic RL (SAC, TD3).

Both algorithms are actor-critic methods that learn from a replay buffer of
past transitions rather than fresh on-policy rollouts (PPO's approach) — this
is what makes them far more sample-efficient for continuous robot control.
They share: a replay buffer, target networks updated by Polyak averaging, and
an MLP builder. Kept here once instead of duplicated in both files.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ReplayBuffer:
    """Fixed-size circular buffer stored directly on the target device (CPU or GPU).

    All tensors are pre-allocated once on the device at construction time.
    ``sample()`` returns GPU tensors via pure on-device indexing — zero PCIe
    copies, zero CUDA memory allocations per update.  This eliminates the
    dominant overhead (5 × numpy→GPU copies + allocations per _update call)
    that makes naive CPU-buffer implementations bottleneck at ~1.8 ms/update
    even with a tiny network.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int,
                 device: str = "cpu") -> None:
        import torch as _torch  # lazy — ReplayBuffer is importable without torch
        self._torch = _torch
        self.capacity = int(capacity)
        self.device = device
        kw = dict(dtype=_torch.float32, device=device)
        # Pre-allocate all storage on device once.
        self._obs      = _torch.zeros(self.capacity, obs_dim,  **kw)
        self._next_obs = _torch.zeros(self.capacity, obs_dim,  **kw)
        self._act      = _torch.zeros(self.capacity, act_dim,  **kw)
        self._rew      = _torch.zeros(self.capacity,           **kw)
        self._done     = _torch.zeros(self.capacity,           **kw)
        self._ptr = 0
        self._size = 0

    def add(self, obs, act, reward, next_obs, done) -> None:
        """Write one transition.  numpy arrays are copied to device once here."""
        t = self._torch
        i = self._ptr
        # from_numpy is zero-copy on CPU side; copy_ handles the H→D transfer.
        self._obs[i].copy_(t.from_numpy(np.asarray(obs,      dtype=np.float32)), non_blocking=True)
        self._act[i].copy_(t.from_numpy(np.asarray(act,      dtype=np.float32)), non_blocking=True)
        self._next_obs[i].copy_(t.from_numpy(np.asarray(next_obs, dtype=np.float32)), non_blocking=True)
        self._rew[i]  = float(reward)
        self._done[i] = float(done)
        self._ptr  = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, obs, act, reward, next_obs, done) -> None:
        """Write ``N`` transitions at once from on-device tensors.

        This is the vectorized-env fast path: ``obs``/``act``/``next_obs`` are
        ``(N, dim)`` tensors and ``reward``/``done`` are ``(N,)``, all already
        on ``self.device``.  The whole batch lands as a single indexed write —
        zero numpy round-trips, zero per-transition Python — which is what lets
        a ``TorchVecEnv`` feed the buffer without a hot-loop bottleneck.
        """
        t = self._torch
        n = int(obs.shape[0])
        idx = (self._ptr + t.arange(n, device=self.device)) % self.capacity
        self._obs[idx] = obs.to(t.float32)
        self._act[idx] = act.to(t.float32)
        self._next_obs[idx] = next_obs.to(t.float32)
        self._rew[idx] = reward.reshape(-1).to(t.float32)
        self._done[idx] = done.reshape(-1).to(t.float32)
        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int, rng=None) -> dict:
        """Return a batch of GPU tensors — pure on-device indexing, no copies."""
        idx = self._torch.randint(0, self._size, (batch_size,), device=self.device)
        return {
            "obs":      self._obs[idx],
            "act":      self._act[idx],
            "reward":   self._rew[idx].unsqueeze(-1),
            "next_obs": self._next_obs[idx],
            "done":     self._done[idx].unsqueeze(-1),
        }


def run_vectorized_training(ctrl, total_timesteps, warmup_steps, batch_size,
                            updates_per_step, torch):
    """Off-policy training loop over a :class:`TorchVecEnv`.

    Shared by SAC and TD3: every iteration advances ``num_envs`` envs in
    parallel and inserts ``num_envs`` transitions in one on-device write, so
    ``total_timesteps`` is consumed ``num_envs`` at a time.  The only
    per-algorithm bits are ``ctrl._vec_action`` (how to act) and
    ``ctrl._update`` (the gradient step) — everything here is tensor-native and
    never touches numpy in the hot path.

    ``updates_per_step`` gradient steps run per env-step iteration; the default
    of 1 keeps the update-to-data ratio low, which matters when each iteration
    already contributes ``num_envs`` fresh transitions.
    """
    env = ctrl.env
    n_envs = env.num_envs
    obs = env.reset()
    steps = 0
    while steps < int(total_timesteps):
        warmup = steps < warmup_steps
        action = ctrl._vec_action(obs, warmup, torch)
        next_obs, reward, done, start_obs = env.step(action)
        ctrl._buffer.add_batch(obs, action, reward, next_obs, done)
        obs = start_obs
        steps += n_envs
        if len(ctrl._buffer) >= batch_size and not warmup:
            for _ in range(int(updates_per_step)):
                ctrl._update(batch_size, torch)
    return ctrl


def build_mlp(nn: Any, sizes: tuple[int, ...], activation, output_activation=None):
    """``sizes = (in, hidden..., out)``."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


def soft_update(target: Any, source: Any, tau: float) -> None:
    """Polyak averaging: ``target <- tau * source + (1 - tau) * target``."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def hard_update(target: Any, source: Any) -> None:
    target.load_state_dict(source.state_dict())


def random_action(action_space: Any, rng: np.random.Generator) -> np.ndarray:
    """Uniform random action from the controller's own seeded RNG.

    Deliberately *not* ``action_space.sample()`` — a gym ``Box`` seeds its own
    RNG from OS entropy unless the caller explicitly calls
    ``action_space.seed(...)``, so warmup exploration built on it would be
    non-reproducible even when the controller itself is seeded.
    """
    low = np.asarray(action_space.low, dtype=np.float64)
    high = np.asarray(action_space.high, dtype=np.float64)
    return rng.uniform(low, high).astype(np.float32)
