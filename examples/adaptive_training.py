"""The adaptive spine — one call, any env, the fastest correct engine.

`Trainer.auto` probes the env and the hardware, picks an execution strategy,
prints *why*, then runs the ordinary SAC/TD3 `learn()` on the right-wrapped env.
The same call adapts across the whole env-reality spectrum — you don't have to
know in advance whether you handed it MJX, a gym.vector env, or one PyBullet
instance.

Run with:  PYTHONPATH=src python3 examples/adaptive_training.py
"""

import numpy as np

from tau_ctrl import Trainer
from tau_ctrl.algorithms.envs import PendulumSwingUp, TorchPendulum


def evaluate(ctrl, episodes: int = 5) -> float:
    env = PendulumSwingUp(max_torque=15.0)
    rs = []
    for _ in range(episodes):
        obs, _ = env.reset()
        tot = 0.0
        for _ in range(200):
            a, _ = ctrl.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            tot += r
            if term or trunc:
                break
        rs.append(tot)
    return float(np.mean(rs))


# --- Case 1: you hand it an already-vectorized on-device env (MJX/Isaac-like) ---
# Trainer detects ON_DEVICE_VECTORIZED and scales updates_per_step automatically.
ctrl = Trainer.auto(
    "sac",
    env=TorchPendulum(num_envs=256, device="cpu", max_torque=15.0, seed=0),
    total_timesteps=80_000,
    seed=0,
    warmup_steps=512,
    batch_size=256,
)
print("case 1 eval return =", round(evaluate(ctrl), 1), "\n")

# --- Case 2: you only have a single, non-batchable env, but can construct copies ---
# Pass a factory (env_fn); Trainer detects SYNC_VEC and replicates it into a batch.
# Swap the lambda for `lambda: gym.make("AntBulletEnv-v0")` and nothing else changes.
ctrl = Trainer.auto(
    "sac",
    env_fn=lambda: PendulumSwingUp(max_torque=15.0),
    total_timesteps=30_000,
    num_envs=8,
    seed=0,
    warmup_steps=256,
    batch_size=256,
)
print("case 2 eval return =", round(evaluate(ctrl), 1))
