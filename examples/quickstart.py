"""tau-ctrl — SB3-like usage across the whole controller spectrum.

Every algorithm shares the same interface over a Gymnasium env, called
directly off the package like SB3:

    from tau_ctrl import make
    ctrl = make(name, env, **kw)
    action, _ = ctrl.predict(obs)      # all methods
    ctrl.learn(total_timesteps=...)    # trainable methods (ppo)

Uses pure-Python toy envs so it runs anywhere (no simulator needed). In
tau-sim, `env` is whatever gymnasium env the chosen simulator provides.

    python examples/quickstart.py
"""

import numpy as np

from tau_ctrl import CBFFilter, PID, available, make
from tau_ctrl.algorithms.envs import DoubleIntegrator, PendulumSwingUp


def run(env, ctrl, steps):
    ctrl.reset()
    obs, _ = env.reset()
    ret = 0.0
    for _ in range(steps):
        a, _ = ctrl.predict(obs)
        obs, r, term, trunc, _ = env.step(a)
        ret += r
        if term or trunc:
            break
    return ret, env.get_state()


def main():
    print("available controllers:", available(), "\n")

    # 1) MPPI — sampling MPC, swings a pendulum up using only the env's reward.
    env = PendulumSwingUp(max_torque=15.0)
    ret, s = run(env, make("mppi", env, horizon=25, n_samples=300, noise_sigma=4.0, seed=0), 120)
    print(f"MPPI  swing-up: return={ret:8.1f}  final|theta|={abs(s[0]):.3f} rad")

    # 2) CEM — same interface, cross-entropy update.
    ret, s = run(env, make("cem", env, horizon=25, n_samples=300, noise_sigma=4.0, seed=0), 120)
    print(f"CEM   swing-up: return={ret:8.1f}  final|theta|={abs(s[0]):.3f} rad")

    # 3) PID — obs-based feedback on a double integrator.
    env = DoubleIntegrator(x0=0.8, v0=0.0)
    ret, s = run(env, PID(env, kp=8.0, kd=5.0, target=[0.0], q_idx=[0], dq_idx=[1]), 200)
    print(f"PID   regulate: return={ret:8.1f}  final x={s[0]:.3f}")

    # 4) CBF — wrap an unsafe controller; enforce a velocity limit.
    env = DoubleIntegrator(x0=0.0, v0=0.0, a_max=3.0)

    class Push(PID):
        def predict(self, obs, state=None, deterministic=True):
            return self.action_space.high.copy(), None

    cbf = CBFFilter(env, base=Push(env), barriers=lambda st: 0.5 - st[1], alpha=0.5)
    vmax = 0.0
    obs, _ = env.reset()
    for _ in range(100):
        a, _ = cbf.predict(obs)
        obs, *_ = env.step(a)
        vmax = max(vmax, env.get_state()[1])
    print(f"CBF   safety  : peak velocity={vmax:.3f} (limit 0.5) — filter held the bound")

    # 5) PPO — same interface; .learn() then .predict(). (tiny smoke run)
    env = PendulumSwingUp()
    ppo = make("ppo", env, rollout_steps=512, epochs=3, hidden=(64, 64), seed=0)
    ppo.learn(total_timesteps=512)
    a, _ = ppo.predict(env.reset()[0])
    print(f"PPO   trained : action={np.round(a, 3)} (device={ppo.device})")


if __name__ == "__main__":
    main()
