"""Make a MuJoCo Gymnasium env *branchable* for the model-based controllers.

The sampling/gradient planners (MPPI/CEM/ICEM/ILQR) and the CBF safety filter
plan by rolling candidate action sequences forward from the current state, which
needs the env to expose ``get_state``/``set_state`` (see
:mod:`tau_ctrl.algorithms.base`). Classic-control envs provide this via
``.state``; MuJoCo envs don't. :class:`MujocoBranchable` adds it by
snapshotting/restoring MuJoCo's *full physics state* (time, ``qpos``, ``qvel``,
actuator activation, plugin state) through ``mj_getState``/``mj_setState`` — the
complete, deterministic state, not just ``qpos``/``qvel``::

    import gymnasium as gym
    from tau_ctrl import make, MujocoBranchable

    env = MujocoBranchable(gym.make("HalfCheetah-v5"))
    ctrl = make("mppi", env, horizon=20, n_samples=200)   # now branchable
    action, _ = ctrl.predict(obs)

Scope
-----
Works on **any** Gymnasium MuJoCo env — the built-in ones (HalfCheetah, Ant,
Hopper, …) and your own ``MujocoEnv`` subclasses — whose complete state is the
physics state. It does **not** magically make a *non*-MuJoCo env branchable:
branching needs the full state, which an arbitrary env (hidden RNG, external
process) can't expose. If a *custom* MuJoCo env also carries Python-side state
that affects dynamics or reward (a goal, a step counter, an action history),
subclass this and extend ``get_state``/``set_state`` to include it.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class MujocoBranchable(gym.Wrapper):
    """Wrap a MuJoCo Gymnasium env so model-based controllers can branch it.

    Adds ``get_state``/``set_state`` backed by MuJoCo's full physics state.
    Transparent otherwise — ``reset``/``step``/spaces pass straight through.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        raw = env.unwrapped
        if not (hasattr(raw, "model") and hasattr(raw, "data")):
            raise TypeError(
                "MujocoBranchable expects a MuJoCo Gymnasium env (env.unwrapped "
                f"with .model/.data), got {type(raw).__name__}. Non-MuJoCo envs "
                "must provide get_state/set_state themselves to be branchable."
            )
        import mujoco  # noqa: PLC0415

        self._mujoco = mujoco
        self._spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self._size = int(mujoco.mj_stateSize(raw.model, self._spec))

    def get_state(self) -> np.ndarray:
        raw = self.env.unwrapped
        arr = np.zeros(self._size, dtype=np.float64)
        self._mujoco.mj_getState(raw.model, raw.data, arr, self._spec)
        return arr

    def set_state(self, state: Any) -> None:
        raw = self.env.unwrapped
        arr = np.ascontiguousarray(state, dtype=np.float64)
        self._mujoco.mj_setState(raw.model, raw.data, arr, self._spec)
        # Recompute derived quantities (contacts, sensors) so the restored state
        # is fully consistent before the next step is planned/taken.
        self._mujoco.mj_forward(raw.model, raw.data)
