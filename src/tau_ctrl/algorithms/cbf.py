"""Control Barrier Function safety filter (discrete-time, model-free).

Wraps any base controller and minimally edits its action so a safety margin
``h(state) >= 0`` is maintained. It enforces the discrete-time CBF condition

    h(next_state) >= (1 - alpha) * h(state),   alpha in (0, 1]

Since we make no simulator assumptions, the one-step effect of an action on
``h`` is measured by *probing* the branchable env (set_state -> step -> read
h). The barrier is linearized in the action via finite differences, giving a
single-inequality QP whose solution is a closed-form projection of the nominal
action. This keeps it fully simulator-agnostic and cheap.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .base import BaseController, ControllerManifest, get_state, register, set_state

Barrier = Callable[[Any], float]  # state -> margin (>= 0 is safe)


@register("cbf")
class CBFFilter(BaseController):
    """Safety filter: ``u = argmin ||u - u_nom||^2 s.t. discrete-CBF holds``."""

    manifest = ControllerManifest(
        name="cbf", family="safety", requires_branching=True,
    )

    def __init__(
        self,
        env: Any,
        base: BaseController,
        barriers: Barrier | Sequence[Barrier],
        alpha: float = 0.5,
        fd_eps: float = 1e-3,
        passes: int = 3,
        **kw: Any,
    ) -> None:
        self.base = base
        self.barriers: list[Barrier] = list(barriers) if isinstance(barriers, (list, tuple)) else [barriers]
        self.alpha = float(alpha)
        self.fd_eps = float(fd_eps)
        self.passes = int(passes)
        super().__init__(env, **kw)

    def reset(self) -> None:
        self.base.reset()

    def _h_next(self, state0: Any, u: np.ndarray, h: Barrier) -> float:
        set_state(self.env, state0)
        self.env.step(u)
        val = h(get_state(self.env))
        return float(val)

    def predict(self, obs, state=None, deterministic: bool = True):
        u_nom, _ = self.base.predict(obs, state, deterministic)
        u = np.array(u_nom, dtype=float).ravel()
        state0 = get_state(self.env)

        for _ in range(self.passes):
            adjusted = False
            for h in self.barriers:
                h0 = float(h(state0))
                margin = self._h_next(state0, u, h) - (1.0 - self.alpha) * h0
                if margin >= 0:
                    continue
                # Linearize h_next in u via finite differences: g = ∂h_next/∂u.
                # Perturb *inward* per dimension so that when u sits on an
                # actuator bound the probe isn't clipped away (which would
                # falsely read a zero gradient).
                g = np.zeros_like(u)
                base_val = self._h_next(state0, u, h)
                sp = self.action_space
                hi = getattr(sp, "high", None)
                for i in range(u.size):
                    step = self.fd_eps
                    if hi is not None and u[i] + step > hi[i]:
                        step = -self.fd_eps
                    up = u.copy()
                    up[i] += step
                    g[i] = (self._h_next(state0, up, h) - base_val) / step
                denom = float(g @ g)
                if denom < 1e-12:
                    continue  # action has no first-order effect on this barrier
                # Project: min ||Δ||^2 s.t. gᵀΔ + margin >= 0  ->  Δ = -margin g/‖g‖²
                u = u + (-margin) * g / denom
                u = self._clip_action(u)
                adjusted = True
            if not adjusted:
                break

        set_state(self.env, state0)  # restore the real env
        return self._clip_action(u), None
