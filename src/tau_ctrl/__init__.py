"""
tau-ctrl — a simulator-agnostic controller/algorithm framework.

Call algorithms directly off the package:

    from tau_ctrl import make, PID, MPPI, CEM, ICEM, ILQR, CBFFilter, PPO, SAC, TD3

    ctrl = make("mppi", env, horizon=25, n_samples=200)   # or PID(env), PPO(env), ...
    action, _ = ctrl.predict(obs)                          # returns action, next_state
    ctrl.learn(total_timesteps=100_000)                    # trainable methods (ppo/sac/td3)

Every algorithm shares one interface (``predict``/``learn``/``save``/``load``)
over a plain ``gymnasium.Env`` — feedback (PID), sampling-based MPC (MPPI,
CEM, ICEM), gradient-based MPC (ILQR), a safety filter (CBF), on-policy RL
(PPO), and off-policy RL (SAC, TD3 — replay-buffer based, far more
sample-efficient than PPO for continuous control).

Model-based methods (MPPI/CEM/ICEM/ILQR/CBF) need the env to be *branchable*
— to expose ``get_state()``/``set_state()``. Feedback and RL work on any gym
env. :mod:`tau_ctrl.tuning` adds Bayesian/genetic auto-tuning of any gains.
"""

__version__ = "0.1.0"

from .algorithms import (
    CBFFilter,
    CEM,
    ICEM,
    ILQR,
    MPPI,
    PID,
    PPO,
    SAC,
    TD3,
    BaseController,
    ControllerManifest,
    Trainer,
    available,
    get_state,
    is_branchable,
    make,
    probe_env,
    probe_hardware,
    register,
    resolve_device,
    set_state,
)
from .tuning import AutoTuner

__all__ = [
    "make",
    "register",
    "available",
    "is_branchable",
    "get_state",
    "set_state",
    "resolve_device",
    "BaseController",
    "ControllerManifest",
    "PID",
    "MPPI",
    "CEM",
    "ICEM",
    "ILQR",
    "CBFFilter",
    "PPO",
    "SAC",
    "TD3",
    "Trainer",
    "probe_env",
    "probe_hardware",
    "AutoTuner",
    "algorithms",
    "tuning",
]
