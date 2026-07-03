"""Obs-based PID / PD feedback controller."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import BaseController, ControllerManifest, register


@register("pid")
class PID(BaseController):
    """Independent PID over selected observation indices.

    ``tau = kp*(target - q) + kd*(dtarget - dq) + ki*∫(target - q)``

    ``q_idx`` / ``dq_idx`` locate positions and velocities inside the obs
    vector (simulator-agnostic — the caller says where they are). If omitted,
    assumes ``obs = [q(action_dim), dq(action_dim), ...]``.
    """

    manifest = ControllerManifest(name="pid", family="feedback")

    def __init__(
        self,
        env: Any,
        kp: float | np.ndarray = 10.0,
        kd: float | np.ndarray = 1.0,
        ki: float | np.ndarray = 0.0,
        target: Optional[np.ndarray] = None,
        q_idx: Optional[list[int]] = None,
        dq_idx: Optional[list[int]] = None,
        integral_clip: float = 10.0,
        **kw: Any,
    ) -> None:
        self._kp, self._kd, self._ki = kp, kd, ki
        self._target = target
        self._q_idx = q_idx
        self._dq_idx = dq_idx
        self._integral_clip = integral_clip
        super().__init__(env, **kw)

    def _setup(self) -> None:
        n = self.action_dim
        self.q_idx = np.asarray(self._q_idx if self._q_idx is not None else range(n), dtype=int)
        self.dq_idx = np.asarray(
            self._dq_idx if self._dq_idx is not None else range(n, 2 * n), dtype=int
        )
        self.kp = np.broadcast_to(np.atleast_1d(self._kp).astype(float), (len(self.q_idx),)).copy()
        self.kd = np.broadcast_to(np.atleast_1d(self._kd).astype(float), (len(self.q_idx),)).copy()
        self.ki = np.broadcast_to(np.atleast_1d(self._ki).astype(float), (len(self.q_idx),)).copy()
        self.target = (
            np.asarray(self._target, dtype=float)
            if self._target is not None
            else np.zeros(len(self.q_idx))
        )
        self._integral = np.zeros(len(self.q_idx))

    def reset(self) -> None:
        self._integral[:] = 0.0

    def predict(self, obs, state=None, deterministic: bool = True):
        obs = np.asarray(obs, dtype=float).ravel()
        q = obs[self.q_idx]
        dq = obs[self.dq_idx] if len(self.dq_idx) else np.zeros_like(q)
        err = self.target - q
        self._integral = np.clip(self._integral + err, -self._integral_clip, self._integral_clip)
        u = self.kp * err + self.kd * (-dq) + self.ki * self._integral
        return self._clip_action(u), None
