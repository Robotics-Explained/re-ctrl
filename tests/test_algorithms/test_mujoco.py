"""MujocoBranchable: make MuJoCo envs branchable for the model-based planners.

Covers a built-in env (HalfCheetah), a *custom* MujocoEnv subclass, exact
state round-tripping, composition with a gym wrapper, and rejection of
non-MuJoCo envs. All skipped if MuJoCo isn't installed.
"""

import numpy as np
import pytest

pytest.importorskip("mujoco")
gym = pytest.importorskip("gymnasium")

from tau_ctrl import MujocoBranchable, get_state, is_branchable, make, set_state  # noqa: E402


def _halfcheetah():
    return MujocoBranchable(gym.make("HalfCheetah-v5"))


def test_mujoco_env_is_branchable_once_wrapped():
    env = _halfcheetah()
    env.reset(seed=0)
    assert is_branchable(env)
    assert get_state(env).ndim == 1


def test_state_roundtrip_is_exact():
    env = _halfcheetah()
    env.reset(seed=0)
    s0 = get_state(env)
    for _ in range(5):
        env.step(env.action_space.sample())
    set_state(env, s0)
    np.testing.assert_allclose(get_state(env), s0)


def test_composes_with_gym_wrapper():
    from gymnasium.wrappers import TimeLimit

    env = MujocoBranchable(TimeLimit(gym.make("HalfCheetah-v5"), 1000))
    env.reset(seed=0)
    assert is_branchable(env)  # resolved through the wrapper stack


@pytest.mark.parametrize("method", ["mppi", "cem", "icem", "ilqr"])
def test_planner_beats_random_on_mujoco(method):
    steps = 40
    kw = dict(horizon=8) if method == "ilqr" else dict(horizon=8, n_samples=24)
    env = _halfcheetah()
    ctrl = make(method, env, **kw)
    ctrl.reset()
    obs, _ = env.reset(seed=0)
    planned = 0.0
    for _ in range(steps):
        a, _ = ctrl.predict(obs)
        obs, r, *_ = env.step(a)
        planned += r
    rnd_env = gym.make("HalfCheetah-v5")
    o, _ = rnd_env.reset(seed=0)
    random_ret = 0.0
    for _ in range(steps):
        o, r, *_ = rnd_env.step(rnd_env.action_space.sample())
        random_ret += r
    assert planned > random_ret, f"{method}: {planned:.1f} !> random {random_ret:.1f}"


def test_rejects_non_mujoco_env():
    with pytest.raises(TypeError, match="MuJoCo"):
        MujocoBranchable(gym.make("Pendulum-v1"))


# --- a *custom* MujocoEnv subclass works too (not just the built-ins) ---------

_POINT_XML = """
<mujoco>
  <option timestep="0.02"/>
  <worldbody>
    <body name="ball" pos="0 0 0">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="slide" gear="1" ctrlrange="-1 1"/></actuator>
</mujoco>
"""


def _make_custom_env(tmp_path):
    from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

    xml = tmp_path / "point.xml"
    xml.write_text(_POINT_XML)

    class PointEnv(MujocoEnv):
        metadata = {"render_modes": [], "render_fps": 25}  # dt = 0.02 * frame_skip 2

        def __init__(self):
            obs_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float64)
            super().__init__(str(xml), 2, observation_space=obs_space)

        def _get_obs(self):
            return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float64)

        def step(self, action):
            self.do_simulation(action, self.frame_skip)
            obs = self._get_obs()
            return obs, float(-(obs[0] ** 2)), False, False, {}

        def reset_model(self):
            self.set_state(self.init_qpos, self.init_qvel)
            return self._get_obs()

    return PointEnv()


def test_custom_mujoco_env_is_branchable(tmp_path):
    env = MujocoBranchable(_make_custom_env(tmp_path))
    env.reset(seed=0)
    assert is_branchable(env)
    s0 = get_state(env)
    env.step(np.array([1.0], dtype=np.float32))
    set_state(env, s0)
    np.testing.assert_allclose(get_state(env), s0)
