"""tau-ctrl's simulator-agnostic controller/algorithm framework.

One interface for the whole controller spectrum over a Gymnasium env:

    from tau_ctrl import make, PID, MPPI, CEM, ICEM, ILQR, CBFFilter, PPO, SAC, TD3

    ctrl = make("mppi", env, horizon=25, n_samples=200)   # or PID(env), PPO(env), ...
    action, _ = ctrl.predict(obs)                          # returns action, next_state
    ctrl.learn(total_timesteps=100_000)                    # for trainable ones (ppo/sac/td3)

Design goals:
- **Simulator-agnostic:** talks only to the gym API (reset/step/spaces). No
  MuJoCo/MJX assumptions — works with whatever simulator hands it an env.
- **GPU-optional:** torch device is auto-selected for learned methods; sampling
  planners parallelise across a vectorized env when one is supplied.
- **Unified:** feedback (PID), sampling-MPC (MPPI/CEM/ICEM), gradient-based
  MPC (ILQR), safety (CBF), and RL (PPO on-policy; SAC/TD3 off-policy) all
  share ``predict``/``learn``/``save``/``load``.

Model-based methods (MPPI/CEM/ICEM/ILQR/CBF) need the env to be *branchable*
— to expose ``get_state()``/``set_state()``. Feedback and RL work on any gym
env.
"""

from .base import (
    BaseController,
    BranchingNotSupported,
    ControllerManifest,
    available,
    get_state,
    is_branchable,
    make,
    register,
    resolve_device,
    set_state,
)
from .cbf import CBFFilter
from .ilqr import ILQR
from .mppi import CEM, ICEM, MPPI
from .mujoco import MujocoBranchable
from .pid import PID
from .ppo import PPO
from .sac import SAC
from .adapters import GymVectorAdapter, SyncTorchVecEnv, jax_to_torch
from .strategy import (
    EnvCaps,
    HardwareCaps,
    Plan,
    Strategy,
    Trainer,
    probe_env,
    probe_hardware,
    select_strategy,
)
from .td3 import TD3
from .vec_env import TorchPendulum, TorchVecEnv

__all__ = [
    "BaseController",
    "ControllerManifest",
    "make",
    "register",
    "available",
    "is_branchable",
    "get_state",
    "set_state",
    "BranchingNotSupported",
    "resolve_device",
    "PID",
    "MPPI",
    "CEM",
    "ICEM",
    "ILQR",
    "CBFFilter",
    "MujocoBranchable",
    "PPO",
    "SAC",
    "TD3",
    "TorchVecEnv",
    "TorchPendulum",
    "GymVectorAdapter",
    "SyncTorchVecEnv",
    "jax_to_torch",
    "Trainer",
    "Strategy",
    "Plan",
    "EnvCaps",
    "HardwareCaps",
    "probe_env",
    "probe_hardware",
    "select_strategy",
]
