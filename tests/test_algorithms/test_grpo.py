"""GRPO — critic-free policy gradient that branches a group of candidate
actions from the live state (see src/tau_ctrl/algorithms/grpo.py)."""

import gymnasium as gym
import numpy as np
import pytest

from tau_ctrl import make
from tau_ctrl.algorithms.base import BranchingNotSupported
from tau_ctrl.algorithms.envs import PendulumSwingUp


def test_grpo_manifest_requires_branching():
    pytest.importorskip("torch")
    env = PendulumSwingUp()
    ctrl = make("grpo", env, group_size=4)
    assert ctrl.manifest.requires_branching
    assert ctrl.manifest.family == "rl"


class _NonBranchable(gym.Env):
    """A plain gym env with no get_state/set_state and no .state — GRPO needs
    branching (unlike PPO/SAC/TD3), so it must refuse this env."""

    observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, np.float32), {}

    def step(self, action):
        return np.zeros(2, np.float32), 0.0, False, False, {}


def test_grpo_rejects_non_branchable_env():
    pytest.importorskip("torch")
    ctrl = make("grpo", _NonBranchable(), group_size=4, rollout_steps=4)
    with pytest.raises(BranchingNotSupported):
        ctrl.learn(total_timesteps=8)


def test_grpo_learns_swing_up():
    pytest.importorskip("torch")
    env = PendulumSwingUp(max_torque=15.0)
    ctrl = make(
        "grpo", env, hidden=(32, 32), group_size=8, group_horizon=4,
        rollout_steps=64, minibatch=32, seed=0,
    )

    def eval_return(steps=150, seed=1):
        obs, _ = env.reset(seed=seed)
        total = 0.0
        for _ in range(steps):
            a, _ = ctrl.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
        return total

    before = eval_return()
    ctrl.learn(total_timesteps=4000)
    after = eval_return()
    assert after > before, f"GRPO did not improve: before={before:.1f} after={after:.1f}"


def test_grpo_save_load(tmp_path):
    pytest.importorskip("torch")
    from tau_ctrl import GRPO

    env = PendulumSwingUp()
    ctrl = make("grpo", env, hidden=(16, 16), group_size=4, rollout_steps=16, seed=0)
    ctrl.learn(total_timesteps=64)
    obs, _ = env.reset()
    a, _ = ctrl.predict(obs)
    assert a.shape == env.action_space.shape
    assert np.all(a >= env.action_space.low - 1e-6) and np.all(a <= env.action_space.high + 1e-6)

    p = tmp_path / "grpo.pkl"
    ctrl.save(p)
    ctrl2 = GRPO.load(p, env, hidden=(16, 16), group_size=4, rollout_steps=16)
    a2, _ = ctrl2.predict(obs)
    np.testing.assert_allclose(a, a2, atol=1e-5)
