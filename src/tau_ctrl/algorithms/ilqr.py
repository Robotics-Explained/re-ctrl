"""iLQR — gradient-based trajectory optimization (a different family from
MPPI/CEM/ICEM's sampling-based search).

Where the sampling-MPC family searches by rolling out many random action
sequences, iLQR alternates a **backward pass** (locally linearize the
dynamics and quadratically approximate the cost along the current nominal
trajectory, solve the resulting LQR problem for a feedback law) with a
**forward pass** (apply that feedback law, with a backtracking line search on
the feedforward term). It converges in far fewer trajectory evaluations when
the dynamics are reasonably smooth, at the cost of being more sensitive to
strong nonlinearities/discontinuities than sampling-based search.

Simulator-agnostic like the rest of :mod:`tau_ctrl.algorithms`: since the env
is an opaque ``step`` function, the dynamics Jacobians and cost derivatives
are estimated by finite differences on a branchable env (``get_state``/
``set_state``), the same contract MPPI/CEM/CBF already rely on. This is a
lightweight iLQR variant: cost curvature is approximated diagonally and
cross state-action curvature is dropped (a standard, cheap simplification —
full analytic/autodiff Hessians would need a differentiable simulator).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import BaseController, ControllerManifest, get_state, register, set_state


@register("ilqr")
class ILQR(BaseController):
    manifest = ControllerManifest(
        name="ilqr", family="gradient_mpc", requires_branching=True, uses_reward=True,
    )

    def __init__(
        self,
        env: Any,
        horizon: int = 20,
        iterations: int = 5,
        fd_eps: float = 1e-3,
        reg: float = 1e-2,
        line_search_steps: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.0),
        **kw: Any,
    ) -> None:
        self.horizon = int(horizon)
        self.iterations = int(iterations)
        self.fd_eps = float(fd_eps)
        self.reg = float(reg)
        self.line_search_steps = line_search_steps
        super().__init__(env, **kw)

    def _setup(self) -> None:
        self.nu = self.action_dim
        state0 = get_state(self.env)
        self.nx = int(np.asarray(state0).size)
        self._u_nom = np.zeros((self.horizon, self.nu))
        # Probe/roll out against the *unwrapped* env (see MPPI): stepping the
        # wrapped env would trip TimeLimit truncation mid-rollout (producing a
        # short trajectory and an IndexError in the backward pass) and corrupt
        # the live episode's wrapper state.
        self._rollout_env = getattr(self.env, "unwrapped", self.env)

    def reset(self) -> None:
        self._u_nom[:] = 0.0

    # ------------------------------------------------------------------
    # Finite-difference probes on the branchable env
    # ------------------------------------------------------------------
    def _probe(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, float]:
        """One step from (x, u): returns (next_state, reward)."""
        set_state(self.env, x)
        u = self._clip_action(u)
        _, r, _, _, _ = self._rollout_env.step(u)
        return np.asarray(get_state(self.env), dtype=float), float(r)

    def _linearize(self, x: np.ndarray, u: np.ndarray):
        """Central-difference A = df/dx, B = df/du, and cost gradient l_x, l_u
        (l = -reward), plus a diagonal Gauss-Newton approximation of l_xx/l_uu.
        """
        eps = self.fd_eps
        nx, nu = self.nx, self.nu
        A = np.zeros((nx, nx))
        B = np.zeros((nx, nu))
        l_x = np.zeros(nx)
        l_u = np.zeros(nu)
        l_xx = np.zeros(nx)
        l_uu = np.zeros(nu)

        x0, r0 = self._probe(x, u)

        for i in range(nx):
            dx = np.zeros(nx); dx[i] = eps
            xp, rp = self._probe(x + dx, u)
            xm, rm = self._probe(x - dx, u)
            A[:, i] = (xp - xm) / (2 * eps)
            l_x[i] = -(rp - rm) / (2 * eps)
            l_xx[i] = -(rp - 2 * r0 + rm) / (eps ** 2)

        for j in range(nu):
            du = np.zeros(nu); du[j] = eps
            xp, rp = self._probe(x, u + du)
            xm, rm = self._probe(x, u - du)
            B[:, j] = (xp - xm) / (2 * eps)
            l_u[j] = -(rp - rm) / (2 * eps)
            l_uu[j] = -(rp - 2 * r0 + rm) / (eps ** 2)

        set_state(self.env, x)  # leave env as found
        return A, B, l_x, l_u, np.diag(np.maximum(l_xx, 0.0)), np.diag(np.maximum(l_uu, 0.0) + self.reg)

    # ------------------------------------------------------------------
    def _rollout(self, state0: np.ndarray, u_seq: np.ndarray):
        xs = [state0.copy()]
        total_cost = 0.0
        x = state0.copy()
        for t in range(self.horizon):
            set_state(self.env, x)
            u = self._clip_action(u_seq[t])
            _, r, term, trunc, _ = self._rollout_env.step(u)
            x = np.asarray(get_state(self.env), dtype=float)
            xs.append(x.copy())
            total_cost += -float(r)
            if term or trunc:
                break
        set_state(self.env, state0)
        return xs, total_cost

    def _backward_pass(self, xs: list[np.ndarray], u_seq: np.ndarray):
        H = len(u_seq)
        V_x = np.zeros(self.nx)
        V_xx = np.zeros((self.nx, self.nx))
        K = [None] * H
        k = [None] * H
        for t in reversed(range(H)):
            A, B, l_x, l_u, l_xx, l_uu = self._linearize(xs[t], u_seq[t])
            Q_x = l_x + A.T @ V_x
            Q_u = l_u + B.T @ V_x
            Q_xx = l_xx + A.T @ V_xx @ A
            Q_uu = l_uu + B.T @ V_xx @ B
            Q_ux = B.T @ V_xx @ A

            try:
                Q_uu_inv = np.linalg.inv(Q_uu)
            except np.linalg.LinAlgError:
                Q_uu_inv = np.linalg.pinv(Q_uu)

            K[t] = -Q_uu_inv @ Q_ux
            k[t] = -Q_uu_inv @ Q_u

            V_x = Q_x + K[t].T @ Q_uu @ k[t] + K[t].T @ Q_u + Q_ux.T @ k[t]
            V_xx = Q_xx + K[t].T @ Q_uu @ K[t] + K[t].T @ Q_ux + Q_ux.T @ K[t]
            V_xx = 0.5 * (V_xx + V_xx.T)
        return K, k

    def predict(self, obs, state=None, deterministic: bool = True):
        state0 = get_state(self.env)
        u_seq = self._u_nom.copy()
        xs, best_cost = self._rollout(state0, u_seq)

        for _ in range(self.iterations):
            # A rollout can terminate before the full horizon (a genuinely
            # terminating env, e.g. the ant falling), leaving fewer states than
            # controls. Only optimize over the segment we actually have a
            # trajectory for; controls past it stay at their nominal value.
            H = min(len(u_seq), len(xs) - 1)
            if H <= 0:
                break
            K, k = self._backward_pass(xs[:H], u_seq[:H])
            improved = False
            for alpha in self.line_search_steps:
                if alpha == 0.0:
                    break  # no-op fallback: keep current nominal
                u_try = u_seq.copy()
                x = state0.copy()
                for t in range(H):
                    u_try[t] = self._clip_action(
                        u_seq[t] + alpha * k[t] + K[t] @ (x - xs[t])
                    )
                    x = self._probe(x, u_try[t])[0]
                xs_try, cost_try = self._rollout(state0, u_try)
                if cost_try < best_cost:
                    u_seq, xs, best_cost = u_try, xs_try, cost_try
                    improved = True
                    break
            if not improved:
                break

        self._u_nom = u_seq
        action = self._clip_action(self._u_nom[0])
        # Receding horizon: shift the plan forward one step.
        self._u_nom = np.roll(self._u_nom, -1, axis=0)
        self._u_nom[-1] = 0.0
        return action, None
