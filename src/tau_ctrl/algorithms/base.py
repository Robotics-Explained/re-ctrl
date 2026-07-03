"""Core abstractions for tau-ctrl's algorithm framework — an SB3-like,
simulator-agnostic controller framework over a Gymnasium environment.

Every method (PID, MPPI, CEM-MPC, CBF safety filter, PPO, ...) implements the
same :class:`BaseController` interface — ``predict(obs) -> action`` like SB3,
plus an optional ``learn()`` for the ones that train. Nothing here imports a
simulator: controllers talk only to the ``gymnasium.Env`` API. Methods that
need to roll the dynamics forward (MPPI, CEM, CBF) do so through the optional
*branching* capability (:func:`get_state` / :func:`set_state`), which any env
can provide; nothing is MuJoCo- or MJX-specific.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type["BaseController"]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def make(name: str, env: Any, **kwargs: Any) -> "BaseController":
    """Construct a controller by name, SB3-style: ``make("mppi", env, ...)``."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown controller {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](env, **kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class ControllerManifest:
    name: str
    family: str                     # "feedback" | "sampling_mpc" | "safety" | "rl"
    requires_branching: bool = False  # needs get_state/set_state for rollouts
    uses_reward: bool = False          # uses the env's reward as its objective
    trainable: bool = False            # has a meaningful learn()
    gpu_capable: bool = False
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Branching capability (simulator-agnostic)
# ---------------------------------------------------------------------------

class BranchingNotSupported(RuntimeError):
    pass


def get_state(env: Any) -> Any:
    """Snapshot the env's dynamical state so a rollout can branch from it.

    Resolution order (no simulator assumptions): an explicit ``get_state`` on
    the (unwrapped) env, else classic-control ``.state``. Simulators that want
    model-based control just expose ``get_state``/``set_state``.
    """
    u = getattr(env, "unwrapped", env)
    if hasattr(u, "get_state"):
        return u.get_state()
    if hasattr(u, "state") and u.state is not None:
        return np.array(u.state, dtype=float)
    raise BranchingNotSupported(
        f"{type(u).__name__} exposes no get_state()/.state; model-based methods "
        "(MPPI/CEM/CBF) need a branchable env."
    )


def set_state(env: Any, state: Any) -> None:
    u = getattr(env, "unwrapped", env)
    if hasattr(u, "set_state"):
        u.set_state(state)
        return
    if hasattr(u, "state"):
        u.state = np.array(state, dtype=float)
        return
    raise BranchingNotSupported(f"{type(u).__name__} exposes no set_state().")


def is_branchable(env: Any) -> bool:
    try:
        get_state(env)
        return True
    except BranchingNotSupported:
        return False


# ---------------------------------------------------------------------------
# Device (only relevant for the learned/torch methods)
# ---------------------------------------------------------------------------

def resolve_device(pref: str = "auto") -> str:
    if pref not in ("auto", "cuda", "cpu"):
        return pref
    if pref == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Base controller
# ---------------------------------------------------------------------------

class BaseController(ABC):
    """SB3-like controller: ``predict`` always; ``learn`` when trainable."""

    name: str = "base"
    manifest: ControllerManifest

    def __init__(self, env: Any, device: str = "auto", seed: Optional[int] = None) -> None:
        self.env = env
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)
        self.device = resolve_device(device)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            # Seed torch's global RNG too, before _setup() builds any network —
            # weight init and stochastic-policy sampling (torch.randn_like) use
            # torch's own RNG, not self.rng, so this is required for
            # reproducibility in any torch-based controller (PPO/SAC/TD3/...).
            try:
                import torch

                torch.manual_seed(seed)
            except ImportError:
                pass
        self._setup()

    # Subclasses override _setup / predict; learn is optional.
    def _setup(self) -> None:  # noqa: B027
        pass

    @abstractmethod
    def predict(
        self, obs: Any, state: Any = None, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]:
        """Return ``(action, next_internal_state)`` — SB3 signature."""

    def learn(self, total_timesteps: int = 0, **kwargs: Any) -> "BaseController":
        """Train the controller. Analytic controllers are no-ops."""
        return self

    def reset(self) -> None:
        """Reset any internal controller state (call at episode start)."""

    # --- helpers ---------------------------------------------------------
    def _clip_action(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        sp = self.action_space
        if sp is not None and getattr(sp, "low", None) is not None:
            a = np.clip(a, sp.low, sp.high)
        return a

    @property
    def action_dim(self) -> int:
        return int(np.prod(self.action_space.shape))

    # --- persistence -----------------------------------------------------
    def _state_dict(self) -> dict:
        """Override to persist trained parameters."""
        return {}

    def _load_state_dict(self, sd: dict) -> None:  # noqa: B027
        pass

    def save(self, path: str | Path) -> None:
        blob = {"name": self.name, "state": self._state_dict()}
        with open(path, "wb") as f:
            pickle.dump(blob, f)

    @classmethod
    def load(cls, path: str | Path, env: Any, **kwargs: Any) -> "BaseController":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        obj = cls(env, **kwargs)
        obj._load_state_dict(blob.get("state", {}))
        return obj
