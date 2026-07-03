"""skrl side of the comparison — same envs/steps/net as the SB3 + tau-ctrl runs.

skrl is torch-native and GPU-capable (like tau-ctrl), so this isolates API
ergonomics and reliability rather than the CPU-vs-GPU gap that SB3 has. Note the
skrl SAC actor needed a manually tanh-bounded mean to avoid divergence on
bounded-action gym envs (see _skrl_helpers) — tau-ctrl/SB3 squash by default.

Run: PYTHONPATH=src:benchmarks python benchmarks/skrl_benchmark.py
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))  # for _skrl_helpers

import numpy as np
import torch
import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

from _skrl_helpers import build_sac, build_td3

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEV}  ({torch.cuda.get_device_name(0) if DEV=='cuda' else 'CPU'})\n")


def eval_agent(agent, env_id, episodes=10, max_steps=1000):
    agent.set_mode("eval")
    ee = gym.make(env_id); rs = []
    for _ in range(episodes):
        o, _ = ee.reset(); tot = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                x = torch.tensor(o, dtype=torch.float32, device=DEV).unsqueeze(0)
                a = agent.act(x, timestep=10**9, timesteps=10**9)[0]
            o, r, te, tr, _ = ee.step(a.cpu().numpy().ravel()); tot += r
            if te or tr: break
        rs.append(tot)
    ee.close()
    return float(np.mean(rs))


def run(env_id, algo, steps, seeds, warmup=100):
    build = {"sac": build_sac, "td3": build_td3}[algo]
    ts, rs = [], []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        env = wrap_env(gym.make(env_id))
        agent = build(env, DEV, warmup)
        t0 = time.perf_counter()
        SequentialTrainer(
            cfg={"timesteps": steps, "headless": True, "disable_progressbar": True},
            env=env, agents=agent,
        ).train()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        rs.append(eval_agent(agent, env_id))
        env.close()
    return float(np.mean(ts)), float(np.mean(rs))


CONFIGS = [
    ("Pendulum-v1",     "sac", 12_000, [0, 1]),
    ("HalfCheetah-v4",  "sac", 30_000, [0]),
    ("Hopper-v4",       "sac", 30_000, [0]),
]
print(f"{'env':<16}{'algo':<5}{'steps':>7} | {'skrl t(s)':>10}{'skrl R':>10}")
print("-" * 52)
for env_id, algo, steps, seeds in CONFIGS:
    t, r = run(env_id, algo, steps, seeds)
    print(f"{env_id:<16}{algo:<5}{steps:>7} | {t:>10.1f}{r:>10.1f}")
print("\nDone.")
