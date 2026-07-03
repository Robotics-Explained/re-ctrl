"""The adaptive spine — probe the env + hardware, pick the fastest *correct*
execution strategy, dispatch. One ``Trainer.auto(...)`` call adapts to whatever
you hand it, so tau-ctrl generalizes past any single env.

SB3 has one execution model (CPU vector env, synchronous collect-then-update)
regardless of what env you give it. The whole speed argument for tau-ctrl is
that wall-clock-to-reward is bounded by the *slower* of collection and update,
and which one dominates depends on the env — so a general framework must detect
the regime and route to the matching engine:

    ON_DEVICE_VECTORIZED  env already batched + on GPU (MJX/Brax via dlpack,
                          Isaac Gym, or a native TorchVecEnv) → 10-100x, the
                          real win; SB3 can't do this at all.
    GYM_VECTOR_ADAPTED    a gymnasium.vector.VectorEnv (numpy/CPU) → batched
                          update on device, collection stays CPU.
    SYNC_VEC              a single non-batchable env replicated N times, stepped
                          in one process → same trainer, batched update.
    SINGLE_ENV           one env you can't replicate (a real robot) → only the
                          update is GPU-accelerated; honest, no false promise.

Not-yet-built engines (true subprocess/async actor-learner for CPU-env-plus-GPU
parallelism) are named in the rationale and the selector degrades to the best
available path instead of silently pretending. Transparency is the feature: every
choice comes with a printable rationale so you know exactly which regime you're in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from .adapters import GymVectorAdapter, SyncTorchVecEnv
from .base import is_branchable, make, resolve_device
from .vec_env import TorchVecEnv


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------

@dataclass
class HardwareCaps:
    device: str          # resolved best torch device
    has_cuda: bool
    n_gpus: int
    n_cpus: int

    def summary(self) -> str:
        gpu = f"{self.n_gpus} GPU" if self.has_cuda else "no GPU"
        return f"device={self.device}, {gpu}, {self.n_cpus} CPU cores"


def probe_hardware(device_pref: str = "auto") -> HardwareCaps:
    has_cuda, n_gpus = False, 0
    try:
        import torch  # noqa: PLC0415

        has_cuda = bool(torch.cuda.is_available())
        n_gpus = torch.cuda.device_count() if has_cuda else 0
    except Exception:
        pass
    return HardwareCaps(
        device=resolve_device(device_pref),
        has_cuda=has_cuda,
        n_gpus=n_gpus,
        n_cpus=os.cpu_count() or 1,
    )


@dataclass
class EnvCaps:
    num_envs: int
    batched: bool
    backend: str          # "torch" | "numpy" | "jax" | "unknown"
    on_device: bool       # step already yields device tensors (no H2D copy)
    branchable: bool
    is_torch_vec: bool
    is_gym_vector: bool
    obs_dim: int
    act_dim: int

    def summary(self) -> str:
        kind = (
            "native TorchVecEnv" if self.is_torch_vec
            else "gym.vector" if self.is_gym_vector
            else "single env"
        )
        return (f"{kind}, num_envs={self.num_envs}, backend={self.backend}, "
                f"on_device={self.on_device}, obs_dim={self.obs_dim}, act_dim={self.act_dim}")


def _space_dim(space) -> int:
    shape = getattr(space, "shape", None)
    return int(np.prod(shape)) if shape else 0


def probe_env(env: Any) -> EnvCaps:
    """Introspect an env once to decide how it can be driven.

    Purely structural — no stepping, no side effects. Recognises native
    :class:`TorchVecEnv`, ``gymnasium.vector.VectorEnv`` (has ``single_*``
    spaces + ``num_envs``), and single gym envs; the model-based branchability
    check is the same seam the MPC controllers use.
    """
    is_torch_vec = isinstance(env, TorchVecEnv)
    num_envs = int(getattr(env, "num_envs", 1) or 1)
    is_gym_vector = (
        not is_torch_vec
        and hasattr(env, "num_envs")
        and hasattr(env, "single_observation_space")
    )
    if is_torch_vec:
        backend, on_device = "torch", str(getattr(env, "device", "cpu")) != "cpu"
    elif is_gym_vector:
        backend, on_device = "numpy", False
    else:
        backend, on_device = "numpy", False

    obs_space = getattr(env, "single_observation_space", None) or getattr(env, "observation_space", None)
    act_space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)

    # Branchability only meaningful for a single env we might replicate/roll out.
    branchable = False
    if not (is_torch_vec or is_gym_vector):
        try:
            branchable = is_branchable(env)
        except Exception:
            branchable = False

    return EnvCaps(
        num_envs=num_envs,
        batched=(num_envs > 1) or is_torch_vec or is_gym_vector,
        backend=backend,
        on_device=on_device,
        branchable=branchable,
        is_torch_vec=is_torch_vec,
        is_gym_vector=is_gym_vector,
        obs_dim=_space_dim(obs_space),
        act_dim=_space_dim(act_space),
    )


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

class Strategy(Enum):
    ON_DEVICE_VECTORIZED = "on_device_vectorized"
    GYM_VECTOR_ADAPTED = "gym_vector_adapted"
    SYNC_VEC = "sync_vec"
    SINGLE_ENV = "single_env"


@dataclass
class Plan:
    strategy: Strategy
    num_envs: int
    device: str
    updates_per_step: int
    rationale: str

    def explain(self) -> str:
        return (
            f"[tau-ctrl] strategy={self.strategy.value}  num_envs={self.num_envs}  "
            f"device={self.device}  updates_per_step={self.updates_per_step}\n"
            f"           {self.rationale}"
        )


def select_strategy(
    ec: EnvCaps,
    hw: HardwareCaps,
    can_replicate: bool,
    num_envs_req: Optional[int] = None,
) -> Plan:
    """Map (env caps, hardware, whether we can spawn copies) → an execution plan.

    Also chooses ``updates_per_step`` up front, because the batched paths have
    the throughput≠learning trap: with many envs and 1 update/step you collect
    fast but barely train. We scale gradient updates with parallelism so the
    default actually converges.
    """
    dev = hw.device

    if ec.is_torch_vec:
        n = ec.num_envs
        ups = max(1, min(n // 16, 32))  # more envs → more updates to keep UTD sane
        why = (
            f"Env is a native on-device TorchVecEnv ({ec.backend}, on_device={ec.on_device}). "
            "Best case: env stepping AND the gradient update run batched on the target "
            "device — the regime where tau-ctrl beats SB3 10-100x (SB3's VecEnv can't). "
            f"Scaling updates_per_step→{ups} so {n} fresh transitions/step actually train."
        )
        return Plan(Strategy.ON_DEVICE_VECTORIZED, n, dev, ups, why)

    if ec.is_gym_vector:
        n = ec.num_envs
        ups = max(1, min(n // 4, 8))
        why = (
            f"Env is a gymnasium.vector.VectorEnv ({n} envs, numpy/CPU). Collection stays "
            "CPU-bound (inherent to gym's vector API), but the replay buffer and gradient "
            "update run batched on device via GymVectorAdapter (one H2D copy/step)."
        )
        return Plan(Strategy.GYM_VECTOR_ADAPTED, n, dev, ups, why)

    # Single env.
    if can_replicate:
        n = num_envs_req or (min(64, 4 * hw.n_cpus) if hw.has_cuda else max(2, hw.n_cpus))
        ups = max(1, min(n // 4, 8))
        async_note = (
            " (A true subprocess/async actor-learner engine would parallelise collection "
            "across cores and overlap it with GPU updates — not yet built, so this SYNC_VEC "
            "path steps the copies in one process: use it for envs that release the GIL, or "
            "when the batched update is the point.)"
        )
        why = (
            f"Single non-batchable env, but replicable → running {n} copies as one batch "
            f"(SyncTorchVecEnv) so the same trainer + batched on-device update apply.{async_note}"
        )
        return Plan(Strategy.SYNC_VEC, n, dev, ups, why)

    why = (
        "Single env, not replicable (e.g. a real robot). Physics can't be parallelised — "
        "only the gradient update is GPU-accelerated. Honest regime: beat SB3 here on "
        "sample efficiency (fewer env steps to reward), not raw throughput. Use SAC/TD3."
    )
    return Plan(Strategy.SINGLE_ENV, 1, dev, 1, why)


# ---------------------------------------------------------------------------
# Trainer — the single adaptive entry point
# ---------------------------------------------------------------------------

class Trainer:
    """One entry point that probes, plans, builds the right env, and trains.

    ``Trainer.auto("sac", env_fn=make_ant, total_timesteps=1_000_000)`` — hand it
    an env (or a zero-arg factory so it can replicate for SYNC_VEC), and it picks
    the strategy for your hardware and runs the ordinary SAC/TD3 ``learn()`` on
    the appropriately-wrapped env. Pass ``env=`` for an already-vectorized env
    (MJX/Isaac/gym.vector) you don't want rebuilt.
    """

    @staticmethod
    def plan(
        env: Any = None,
        env_fn: Optional[Callable[[], Any]] = None,
        device: str = "auto",
        num_envs: Optional[int] = None,
    ) -> tuple[Plan, EnvCaps, HardwareCaps]:
        if env is None and env_fn is None:
            raise ValueError("provide env= or env_fn=")
        hw = probe_hardware(device)
        sample = env if env is not None else env_fn()
        ec = probe_env(sample)
        plan = select_strategy(ec, hw, can_replicate=(env_fn is not None), num_envs_req=num_envs)
        return plan, ec, hw

    @classmethod
    def auto(
        cls,
        algo: str,
        env: Any = None,
        env_fn: Optional[Callable[[], Any]] = None,
        total_timesteps: int = 100_000,
        device: str = "auto",
        num_envs: Optional[int] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        algo_kwargs: Optional[dict] = None,
        **learn_kwargs: Any,
    ):
        plan, ec, hw = cls.plan(env, env_fn, device, num_envs)
        if verbose:
            print(f"[tau-ctrl] hardware: {hw.summary()}")
            print(f"[tau-ctrl] env:      {ec.summary()}")
            print(plan.explain())

        train_env = cls._build_env(plan, env, env_fn)
        ctrl = make(algo, train_env, device=plan.device, seed=seed, **(algo_kwargs or {}))
        # Fill in a convergence-safe updates_per_step for the batched paths unless
        # the caller pinned one — addresses the throughput≠learning trap by default.
        learn_kwargs.setdefault("updates_per_step", plan.updates_per_step)
        if plan.strategy is Strategy.SINGLE_ENV:
            learn_kwargs.pop("updates_per_step", None)  # single-env learn has no such knob
        ctrl.learn(total_timesteps=total_timesteps, **learn_kwargs)
        return ctrl

    @staticmethod
    def _build_env(plan: Plan, env: Any, env_fn: Optional[Callable[[], Any]]):
        if plan.strategy is Strategy.ON_DEVICE_VECTORIZED:
            return env if env is not None else env_fn()
        if plan.strategy is Strategy.GYM_VECTOR_ADAPTED:
            venv = env if env is not None else env_fn()
            return GymVectorAdapter(venv, device=plan.device)
        if plan.strategy is Strategy.SYNC_VEC:
            assert env_fn is not None
            return SyncTorchVecEnv([env_fn for _ in range(plan.num_envs)], device=plan.device)
        return env if env is not None else env_fn()  # SINGLE_ENV
