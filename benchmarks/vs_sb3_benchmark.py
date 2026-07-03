"""tau-ctrl vs Stable-Baselines3 — head-to-head on several env types (GPU).

Three comparisons, honest about what each shows:
  1. Apples-to-apples single env (Pendulum-v1): same algo, same steps, same
     net — isolates core implementation + update speed. Modest expected win.
  2. MuJoCo continuous control (HalfCheetah, Hopper): "different types" — tests
     that tau-ctrl generalizes and surfaces any reliability (reward) gap.
  3. The architectural win: tau-ctrl on a vectorized on-device env (TorchPendulum,
     N envs on GPU) vs SB3 on the single env — wall-clock to reach a reward.

Run with the CUDA-enabled interpreter:
    PYTHONPATH=src /home/tayalmanan28/anaconda3/bin/python scripts/vs_sb3_benchmark.py
"""

import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import numpy as np
import torch
import gymnasium as gym

from tau_ctrl import make
from tau_ctrl.algorithms.envs import TorchPendulum, PendulumSwingUp

from stable_baselines3 import SAC as SB3SAC, TD3 as SB3TD3

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEV}  ({torch.cuda.get_device_name(0) if DEV=='cuda' else 'CPU'})\n")


def eval_gym(env_id, predict_fn, episodes=10, max_steps=1000):
    env = gym.make(env_id)
    rs = []
    for _ in range(episodes):
        obs, _ = env.reset()
        tot = 0.0
        for _ in range(max_steps):
            a = predict_fn(obs)
            obs, r, term, trunc, _ = env.step(a)
            tot += r
            if term or trunc:
                break
        rs.append(tot)
    env.close()
    return float(np.mean(rs)), float(np.std(rs))


def run_pair(env_id, algo, steps, seeds, warmup=100):
    """Return dict of tau-ctrl vs SB3 wall-time and reward, averaged over seeds."""
    TC = {"sac": "sac", "td3": "td3"}[algo]
    SB = {"sac": SB3SAC, "td3": SB3TD3}[algo]
    tc_t, tc_r, sb_t, sb_r = [], [], [], []
    for s in seeds:
        # tau-ctrl
        env = gym.make(env_id)
        c = make(TC, env, seed=s, device=DEV)          # default hidden (256,256)
        t0 = time.perf_counter()
        c.learn(total_timesteps=steps, warmup_steps=warmup, batch_size=256)
        if DEV == "cuda": torch.cuda.synchronize()
        tc_t.append(time.perf_counter() - t0)
        tc_r.append(eval_gym(env_id, lambda o: c.predict(o, deterministic=True)[0])[0])
        env.close()

        # SB3 (same net, same device)
        m = SB(
            "MlpPolicy", gym.make(env_id), seed=s, device=DEV, verbose=0,
            learning_starts=warmup, policy_kwargs=dict(net_arch=[256, 256]),
        )
        t0 = time.perf_counter()
        m.learn(total_timesteps=steps, progress_bar=False)
        if DEV == "cuda": torch.cuda.synchronize()
        sb_t.append(time.perf_counter() - t0)
        sb_r.append(eval_gym(env_id, lambda o: m.predict(o, deterministic=True)[0])[0])
    return dict(
        tc_time=np.mean(tc_t), tc_reward=np.mean(tc_r),
        sb_time=np.mean(sb_t), sb_reward=np.mean(sb_r),
    )


# ── 1 & 2: single-env, apples-to-apples ────────────────────────────────────
CONFIGS = [
    ("Pendulum-v1",     "sac", 12_000, [0, 1]),
    ("Pendulum-v1",     "td3", 12_000, [0, 1]),
    ("HalfCheetah-v4",  "sac", 30_000, [0]),
    ("Hopper-v4",       "sac", 30_000, [0]),
]
print("=" * 78)
print("SINGLE-ENV, APPLES-TO-APPLES  (same algo, steps, net=[256,256], device)")
print("=" * 78)
print(f"{'env':<16}{'algo':<5}{'steps':>7} | {'tau t(s)':>9}{'SB3 t(s)':>9}{'speedup':>8} | "
      f"{'tau R':>9}{'SB3 R':>9}")
print("-" * 78)
for env_id, algo, steps, seeds in CONFIGS:
    r = run_pair(env_id, algo, steps, seeds)
    spd = r["sb_time"] / r["tc_time"] if r["tc_time"] else float("nan")
    print(f"{env_id:<16}{algo:<5}{steps:>7} | {r['tc_time']:>9.1f}{r['sb_time']:>9.1f}"
          f"{spd:>7.2f}x | {r['tc_reward']:>9.1f}{r['sb_reward']:>9.1f}")

# ── 3: architectural win — vectorized on-device vs SB3 single env ───────────
print("\n" + "=" * 78)
print("ARCHITECTURAL WIN  — wall-clock to reach eval return >= -200 on pendulum")
print("=" * 78)

def solve_time_tauctrl_vec(N, target=-200.0, chunk=20_000, budget=400_000):
    env = TorchPendulum(num_envs=N, device=DEV, max_torque=15.0, seed=0)
    c = make("sac", env, hidden=(64, 64), seed=0, device=DEV)
    done_steps = 0
    t0 = time.perf_counter()
    while done_steps < budget:
        c.learn(total_timesteps=chunk, warmup_steps=(2 * N if done_steps == 0 else 0),
                batch_size=512, updates_per_step=16)
        done_steps += chunk
        r, _ = eval_scalar(c)
        if r >= target:
            break
    if DEV == "cuda": torch.cuda.synchronize()
    return time.perf_counter() - t0, done_steps, r

def eval_scalar(c, episodes=5):
    env = PendulumSwingUp(max_torque=15.0); rs = []
    for _ in range(episodes):
        obs, _ = env.reset(); tot = 0.0
        for _ in range(200):
            a, _ = c.predict(obs, deterministic=True)
            obs, rr, term, trunc, _ = env.step(a); tot += rr
            if term or trunc: break
        rs.append(tot)
    return float(np.mean(rs)), float(np.std(rs))

def solve_time_sb3(target=-200.0, chunk=2_000, budget=40_000):
    m = SB3SAC("MlpPolicy", gym.make("Pendulum-v1"), seed=0, device=DEV, verbose=0,
               learning_starts=100, policy_kwargs=dict(net_arch=[64, 64]))
    done = 0; t0 = time.perf_counter()
    while done < budget:
        m.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        done += chunk
        r, _ = eval_gym("Pendulum-v1", lambda o: m.predict(o, deterministic=True)[0], episodes=5)
        if r >= target:
            break
    if DEV == "cuda": torch.cuda.synchronize()
    return time.perf_counter() - t0, done, r

for N in (256, 1024):
    t, steps, r = solve_time_tauctrl_vec(N)
    print(f"tau-ctrl vec N={N:5d}: {t:6.1f}s  env_steps={steps:>7}  final_eval={r:7.1f}")
t, steps, r = solve_time_sb3()
print(f"SB3 single-env    : {t:6.1f}s  env_steps={steps:>7}  final_eval={r:7.1f}")
print("\nDone.")
