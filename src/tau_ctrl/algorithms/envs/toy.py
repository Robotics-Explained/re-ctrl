"""Tiny pure-Python Gymnasium envs for testing/examples.

No simulator dependency (no MuJoCo/MJX) — they exist so tau-ctrl's algorithms
can be tested end-to-end anywhere. Both expose ``get_state``/``set_state`` so
the model-based controllers (MPPI/CEM/CBF) can branch rollouts, exactly as a
real simulator env would once tau-sim adds that capability.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError("tau_ctrl.algorithms requires gymnasium") from e


class PendulumSwingUp(gym.Env):
    """Torque-limited pendulum. angle 0 = upright; reward peaks upright.

    obs = [cos(th), sin(th), thdot]. Reward = -(th^2 + 0.1*thdot^2 + 1e-3*u^2).
    """

    metadata = {"render_modes": []}

    def __init__(self, max_torque: float = 6.0, dt: float = 0.05,
                 g: float = 10.0, m: float = 1.0, l: float = 1.0,
                 start_angle: float = np.pi, max_speed: float = 8.0):
        self.max_torque = float(max_torque)
        self.max_speed = float(max_speed)
        self.dt, self.g, self.m, self.l = dt, g, m, l
        self.start_angle = float(start_angle)
        self.state = np.array([self.start_angle, 0.0])
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-self.max_torque, self.max_torque, (1,), np.float32)

    # --- branching capability ---
    def get_state(self):
        return np.array(self.state, dtype=float)

    def set_state(self, s):
        self.state = np.array(s, dtype=float)

    def _obs(self):
        th, thd = self.state
        return np.array([np.cos(th), np.sin(th), thd], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = np.array([self.start_angle, 0.0])
        return self._obs(), {}

    def step(self, action):
        th, thd = self.state
        u = float(np.clip(np.asarray(action).reshape(-1)[0], -self.max_torque, self.max_torque))
        # angle measured from upright: gravity accelerates away from upright.
        thddot = (self.g / self.l) * np.sin(th) + u / (self.m * self.l ** 2)
        thd = np.clip(thd + thddot * self.dt, -self.max_speed, self.max_speed)
        th = th + thd * self.dt
        th = ((th + np.pi) % (2 * np.pi)) - np.pi  # wrap to [-pi, pi]
        self.state = np.array([th, thd])
        cost = th ** 2 + 0.1 * thd ** 2 + 1e-3 * u ** 2
        return self._obs(), float(-cost), False, False, {}


class DoubleIntegrator(gym.Env):
    """1-D point mass: state=[x, v], action = acceleration (bounded).

    Reward drives x -> 0. Handy for testing a CBF that must keep x <= x_max.
    obs = [x, v].
    """

    metadata = {"render_modes": []}

    def __init__(self, dt: float = 0.05, a_max: float = 3.0, x_max: float = 1.0,
                 x0: float = 0.5, v0: float = 1.5):
        self.dt, self.a_max, self.x_max = dt, float(a_max), float(x_max)
        self.x0, self.v0 = float(x0), float(v0)
        self.state = np.array([x0, v0])
        self.observation_space = spaces.Box(-np.inf, np.inf, (2,), np.float32)
        self.action_space = spaces.Box(-self.a_max, self.a_max, (1,), np.float32)

    def get_state(self):
        return np.array(self.state, dtype=float)

    def set_state(self, s):
        self.state = np.array(s, dtype=float)

    def _obs(self):
        return self.state.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([self.x0, self.v0])
        return self._obs(), {}

    def step(self, action):
        x, v = self.state
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -self.a_max, self.a_max))
        v = v + a * self.dt
        x = x + v * self.dt
        self.state = np.array([x, v])
        reward = -(x ** 2 + 0.1 * v ** 2)
        return self._obs(), float(reward), False, False, {}
