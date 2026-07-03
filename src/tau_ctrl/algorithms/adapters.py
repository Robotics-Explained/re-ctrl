"""Adapters: make *whatever env you actually have* speak the TorchVecEnv
contract, so the same SAC/TD3 vectorized loop runs unchanged.

You rarely own the environment. The batched, tensor-native
:class:`~tau_ctrl.algorithms.vec_env.TorchVecEnv` is the ideal, but real
projects hand you one of three things, and tau-ctrl adapts to all three behind
the same trainer (the loop only needs ``num_envs`` + ``reset`` + ``step`` →
``(next_obs, reward, done, start_obs)``):

1. **Already batched & on-device** — MJX / Brax (JAX) or Isaac Gym (torch).
   Wrap with :func:`jax_to_torch` (zero-copy via dlpack) or, for a torch-native
   batched env, nothing at all — it already fits the contract.

2. **Batched but numpy/CPU** — a ``gymnasium.vector.VectorEnv``.
   :class:`GymVectorAdapter` moves each step's batch to the device with one
   copy. Collection stays CPU-bound, but the replay buffer and gradient update
   are batched and on-device.

3. **A single, non-batchable env** — PyBullet, classic MuJoCo, a real robot.
   You can't turn its dynamics into tensor ops, but you can run ``N`` copies.
   :class:`SyncTorchVecEnv` steps them in a Python loop and presents the batch
   on-device. No GPU *env* parallelism (the env is the bottleneck), but the
   exact same trainer runs and the update still saturates the GPU.

If you truly have one env and can't make copies (a single real robot), just use
the ordinary single-env ``learn()`` path — there the GPU only accelerates the
update, which is honest: you can't parallelize physics you don't control.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .vec_env import TorchVecEnv


def jax_to_torch(x, device: str):
    """Zero-copy JAX array → torch tensor via dlpack (for MJX / Brax on GPU).

    Both frameworks expose ``__dlpack__``; the buffer is shared, so no host
    round-trip and no extra device allocation. Use this inside a thin
    ``TorchVecEnv`` subclass that drives an MJX/Brax ``step`` and hands the
    resulting device arrays straight to the trainer.
    """
    import torch  # noqa: PLC0415

    t = torch.from_dlpack(x)
    return t.to(device) if str(t.device) != str(device) else t


class GymVectorAdapter(TorchVecEnv):
    """Wrap a ``gymnasium.vector.VectorEnv`` (numpy batches) as a TorchVecEnv.

    Each ``step`` moves the batch to the device with a single H→D copy. The env
    still runs on CPU (that's inherent to gym's vector API), but the buffer and
    update are batched on-device — the same win as a native ``TorchVecEnv``
    minus the on-device env stepping.

    Handles gym autoreset: on the step where an env is done, gym returns the
    *reset* observation and stashes the true terminal obs in
    ``info["final_observation"]``. We store the terminal obs as ``next_obs``
    (correct bootstrap target) and carry the reset obs as ``start_obs``.
    """

    def __init__(self, venv: Any, device: str = "cpu") -> None:
        super().__init__(getattr(venv, "num_envs"), device)
        self.venv = venv
        # gym VectorEnv exposes single_* spaces describing one env.
        self.observation_space = getattr(venv, "single_observation_space", None) or venv.observation_space
        self.action_space = getattr(venv, "single_action_space", None) or venv.action_space

    def _to_t(self, arr):
        t = self._torch
        return t.as_tensor(np.asarray(arr, dtype=np.float32), device=self.device)

    def reset(self):
        obs, _ = self.venv.reset()
        return self._to_t(obs)

    def step(self, actions):
        t = self._torch
        a = actions.detach().cpu().numpy()
        obs, reward, terminated, truncated, info = self.venv.step(a)
        done = np.asarray(terminated, dtype=np.float32)
        start_obs = np.asarray(obs, dtype=np.float32).copy()
        next_obs = start_obs.copy()  # true post-dynamics obs

        # Recover terminal observations for envs that just reset.
        finals = info.get("final_observation") if isinstance(info, dict) else None
        if finals is not None:
            for i, fo in enumerate(finals):
                if fo is not None:
                    next_obs[i] = np.asarray(fo, dtype=np.float32)
        return (
            self._to_t(next_obs),
            self._to_t(reward),
            t.as_tensor(done, device=self.device),
            self._to_t(start_obs),
        )


class SyncTorchVecEnv(TorchVecEnv):
    """Run ``N`` copies of a single (non-batchable) gym env as one on-device batch.

    For PyBullet, classic MuJoCo, or any env whose dynamics you can't rewrite as
    tensor ops but can instantiate ``N`` times. The envs step in a plain Python
    loop on CPU — so you get no GPU env parallelism — but the same vectorized
    SAC/TD3 loop runs unchanged and the replay buffer + gradient update stay
    batched on-device.

    ``env_fns`` is a list of zero-arg factories (one per env), matching the SB3
    / gym.vector convention so seeds/configs can differ per env.
    """

    def __init__(self, env_fns: list[Callable[[], Any]], device: str = "cpu") -> None:
        self.envs = [fn() for fn in env_fns]
        super().__init__(len(self.envs), device)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        obs = np.stack([np.asarray(e.reset()[0], dtype=np.float32) for e in self.envs])
        return self._torch.as_tensor(obs, device=self.device)

    def step(self, actions):
        t = self._torch
        a = actions.detach().cpu().numpy()
        n = self.num_envs
        next_obs = np.empty((n, *self.observation_space.shape), dtype=np.float32)
        start_obs = np.empty_like(next_obs)
        reward = np.empty(n, dtype=np.float32)
        done = np.zeros(n, dtype=np.float32)
        for i, e in enumerate(self.envs):
            o, r, term, trunc, _ = e.step(a[i])
            next_obs[i] = np.asarray(o, dtype=np.float32)
            reward[i] = r
            done[i] = float(term)  # truncation resets but doesn't break bootstrap
            if term or trunc:
                o2, _ = e.reset()
                start_obs[i] = np.asarray(o2, dtype=np.float32)
            else:
                start_obs[i] = next_obs[i]
        return (
            t.as_tensor(next_obs, device=self.device),
            t.as_tensor(reward, device=self.device),
            t.as_tensor(done, device=self.device),
            t.as_tensor(start_obs, device=self.device),
        )
