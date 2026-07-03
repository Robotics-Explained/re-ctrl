"""GPU-vectorized RL — the path to real (10-50x) speedup.

Standard vector environments step CPU environments in subprocesses; the policy update is a
tiny GPU (or CPU) op sandwiched between per-step Python. tau-ctrl instead runs
``num_envs`` environments *and* the update entirely on-device, so nothing
crosses the Python/numpy boundary in the hot loop and each step feeds a large
batched update — which is what actually saturates a GPU.

Run with:  PYTHONPATH=src python3 examples/vectorized_rl.py

Set device="cuda" when you have a GPU; the code is identical either way.
"""

import time

import numpy as np

from tau_ctrl import make
from tau_ctrl.algorithms.envs import PendulumSwingUp, TorchPendulum

DEVICE = "cpu"  # switch to "cuda" on a GPU box — no other change needed
NUM_ENVS = 256


def evaluate(ctrl, episodes: int = 5) -> float:
    """Deterministic eval on the scalar env (mean return)."""
    env = PendulumSwingUp(max_torque=15.0)
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        for _ in range(200):
            a, _ = ctrl.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
        returns.append(total)
    return float(np.mean(returns))


def main() -> None:
    # A batched, on-device env: 256 pendulums advanced as one tensor op.
    env = TorchPendulum(num_envs=NUM_ENVS, device=DEVICE, max_torque=15.0, seed=0)
    ctrl = make("sac", env, hidden=(64, 64), seed=0)

    # total_timesteps counts env-steps across ALL envs (NUM_ENVS per iteration).
    # updates_per_step raises the gradient-update-to-data ratio: with NUM_ENVS
    # fresh transitions per iteration you need several updates each iteration to
    # actually learn — throughput alone doesn't train the policy.
    env_steps = 80_000
    t0 = time.perf_counter()
    ctrl.learn(
        total_timesteps=env_steps,
        warmup_steps=2 * NUM_ENVS,
        batch_size=256,
        updates_per_step=16,
    )
    dt = time.perf_counter() - t0

    print(f"device={DEVICE}  num_envs={NUM_ENVS}")
    print(f"env-steps={env_steps}  wall={dt:.1f}s  throughput={env_steps/dt:,.0f} steps/s")
    print(f"eval mean return = {evaluate(ctrl):.1f}   (swing-up solved when > -200)")


if __name__ == "__main__":
    main()
