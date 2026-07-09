"""Regression tests for fixed defects.

Each test pins behaviour that was previously broken:
  * model-based planners corrupting / crashing on a *wrapped* gym env,
  * the hard-coded package version,
  * unhelpful errors on discrete-action or raw vectorized envs.
"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.wrappers import TimeLimit

from tau_ctrl import make
from tau_ctrl.algorithms.envs import PendulumSwingUp


def _wrapped(max_steps=30):
    """A branchable toy env behind gym's TimeLimit — mimics gym.make()."""
    env = TimeLimit(PendulumSwingUp(max_torque=15.0), max_episode_steps=max_steps)
    env.reset(seed=0)
    return env


# --------------------------------------------------------------------------
# BUG-1: model-based planners must roll out against the *unwrapped* env, so a
# single predict() never advances the TimeLimit counter or terminates the live
# episode — and ILQR must not crash when a rollout would trip truncation.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["mppi", "cem", "icem", "ilqr"])
def test_planner_does_not_disturb_wrapped_episode(method):
    env = _wrapped(max_steps=30)
    kw = {} if method == "ilqr" else dict(n_samples=50)
    ctrl = make(method, env, horizon=10, **kw)  # 10*50 rollout steps >> 30-step limit
    obs = env.unwrapped._obs()

    action, _ = ctrl.predict(obs)  # must not raise (ILQR previously IndexError'd)
    assert env._elapsed_steps == 0, "rollout leaked into the wrapper's step counter"
    assert action.shape == env.action_space.shape


@pytest.mark.parametrize("method", ["mppi", "cem", "icem", "ilqr"])
def test_planner_runs_full_wrapped_episode(method):
    env = _wrapped(max_steps=25)
    kw = {} if method == "ilqr" else dict(n_samples=50)
    ctrl = make(method, env, horizon=10, **kw)
    ctrl.reset()
    obs = env.unwrapped._obs()
    steps = 0
    for _ in range(25):
        action, _ = ctrl.predict(obs)
        obs, _, term, trunc, _ = env.step(action)
        steps += 1
        if term or trunc:
            break
    assert steps == 25, f"{method}: episode ended early ({steps} steps) — wrapper corrupted"


# --------------------------------------------------------------------------
# BUG-2: version must reflect the installed package, not a stale literal.
# --------------------------------------------------------------------------

def test_version_matches_installed_metadata():
    import tau_ctrl
    from importlib.metadata import version

    assert tau_ctrl.__version__ == version("tau-ctrl")
    assert tau_ctrl.__version__ != "0.1.0" or version("tau-ctrl") == "0.1.0"


# --------------------------------------------------------------------------
# BUG-4: discrete-action envs are rejected with a clear message (not a deep
# AttributeError). The guard fires before torch is imported.
# --------------------------------------------------------------------------

class _DiscreteEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, np.float32), {}

    def step(self, action):
        return np.zeros(2, np.float32), 0.0, False, False, {}


@pytest.mark.parametrize("method", ["sac", "ppo", "td3"])
def test_discrete_action_space_rejected(method):
    with pytest.raises(TypeError, match="continuous Box action space"):
        make(method, _DiscreteEnv())


# --------------------------------------------------------------------------
# BUG-5: a raw gymnasium.vector.VectorEnv routed through make() fails fast with
# a message pointing at Trainer.auto, instead of crashing inside learn().
# --------------------------------------------------------------------------

def test_raw_vector_env_rejected_with_hint():
    venv = gym.vector.SyncVectorEnv([lambda: PendulumSwingUp() for _ in range(2)])
    with pytest.raises(TypeError, match="Trainer.auto"):
        make("sac", venv)
