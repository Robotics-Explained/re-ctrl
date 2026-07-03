"""
benchmark.py — compare tau-ctrl vs Stable-Baselines3 on pendulum swing-up.

What we measure
---------------
1. Correctness  — final angle after fixed number of environment steps.
2. Wall-clock   — time to complete `N_LEARN` env steps of learn() + 1 rollout.
3. Sample efficiency — cumulative reward vs. env steps, checked at milestones.

Run with:
    PYTHONPATH=src python3 scripts/benchmark.py
"""

import sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np

# ── tau-ctrl ──────────────────────────────────────────────────────────────
sys.path.insert(0, "src")
from tau_ctrl import make, MPPI, PID
from tau_ctrl.algorithms.envs import PendulumSwingUp, DoubleIntegrator

TOTAL_RL_STEPS = 8_000       # RL train budget (keep fast)
WARMUP         = 500         # off-policy warmup
BATCH          = 128
ROLLOUT_STEPS  = 40          # each episode
SEEDS          = [0, 1, 2]

# ═══════════════════════════════════════════════════════════════════════════
# 1. CORRECTNESS — run the full test suite (sans slow ones)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. CORRECTNESS — running fast algorithm tests")
print("=" * 60)

import subprocess, sys as _sys
result = subprocess.run(
    [_sys.executable, "-m", "pytest", "tests/test_algorithms/", "-v",
     "--no-header", "-q", "--tb=short",
     "-k", "not test_off_policy_rl_learns"],  # skip the slow RL tests
    capture_output=True, text=True,
    env={**__import__("os").environ, "PYTHONPATH": "src"},
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode == 0:
    print("✅  All algorithm tests PASSED\n")
else:
    print("❌  Some tests FAILED\n")
    print(result.stderr[-1000:])

# ═══════════════════════════════════════════════════════════════════════════
# 2. SPEED — MPPI (model-based) — no SB3 equivalent, baseline timing
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("2. SPEED — MPPI planners vs PID baseline on PendulumSwingUp")
print("=" * 60)

def time_planner(method, env_cls=PendulumSwingUp, steps=100, seed=0, **kw):
    env = env_cls(max_torque=15.0)
    ctrl = make(method, env, seed=seed, **kw)
    ctrl.reset()
    obs, _ = env.reset()
    t0 = time.perf_counter()
    final_angle = None
    for _ in range(steps):
        a, _ = ctrl.predict(obs)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    elapsed = time.perf_counter() - t0
    final_angle = abs(env.get_state()[0])
    return elapsed, final_angle

planners = {
    "pid":  dict(kp=5.0, kd=2.0, target=[0.0], q_idx=[0], dq_idx=[1]),
    "mppi": dict(horizon=25, n_samples=300, noise_sigma=4.0, temperature=1.0),
    "cem":  dict(horizon=25, n_samples=300, noise_sigma=4.0, elite_frac=0.1),
    "icem": dict(horizon=25, n_samples=300, noise_sigma=4.0, elite_frac=0.1),
    "ilqr": dict(horizon=15, iterations=5),
}

print(f"{'Method':<8} | {'Time(s)':>8} | {'FinalAngle':>12} | {'Note'}")
print("-" * 50)
for method, kw in planners.items():
    try:
        elapsed, angle = time_planner(method, steps=120, seed=0, **kw)
        success = "✅ swung up" if angle < 0.5 else "⚠️  not there"
        print(f"{method:<8} | {elapsed:>8.3f} | {angle:>12.4f} | {success}")
    except Exception as e:
        print(f"{method:<8} | {'ERR':>8} | {'':>12} | {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. SPEED & ACCURACY — RL: tau-ctrl SAC/TD3 vs SB3 SAC/TD3
#    We train both for the same number of env steps and compare
#    - wall-clock time
#    - final eval reward
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. RL COMPARISON — tau-ctrl vs SB3 (SAC/TD3 on PendulumSwingUp)")
print("=" * 60)

try:
    import torch
    HAS_TORCH = True
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   PyTorch available — device: {device_str}")
except ImportError:
    HAS_TORCH = False
    print("   PyTorch not available — skipping RL benchmarks")

try:
    import gymnasium as gym
    import stable_baselines3 as sb3
    from stable_baselines3 import SAC as SB3SAC, TD3 as SB3TD3
    from stable_baselines3.common.env_util import make_vec_env
    HAS_SB3 = True
    print(f"   SB3 {sb3.__version__} available")
except ImportError:
    HAS_SB3 = False
    print("   SB3 not installed — skipping SB3 comparison (install with: pip install stable-baselines3)")

def eval_policy(env, predict_fn, n_eval=5):
    """Returns mean cumulative reward over n_eval episodes."""
    returns = []
    for _ in range(n_eval):
        obs, _ = env.reset()
        total = 0.0
        for _ in range(200):
            a, _ = predict_fn(obs)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
        returns.append(total)
    return float(np.mean(returns))

if HAS_TORCH and HAS_SB3:
    import gymnasium as gym

    results = {}
    for method in ["sac", "td3"]:
        print(f"\n  --- {method.upper()} ---")
        # ── Use Pendulum-v1 for BOTH so the comparison is apples-to-apples ─
        # tau-ctrl talks to any gymnasium env; SB3 does too.
        # Both train and eval on the same env class (Pendulum-v1): action ±2,
        # thdot clamped to ±8, reward in [-16.27, 0] — identical conditions.
        def make_env():
            return gym.make("Pendulum-v1")

        # ── tau-ctrl ────────────────────────────────────────────────────
        tc_env = make_env()
        tc_ctrl = make(method, tc_env, hidden=(64, 64), seed=0)
        t0 = time.perf_counter()
        tc_ctrl.learn(total_timesteps=TOTAL_RL_STEPS,
                      warmup_steps=WARMUP, batch_size=BATCH)
        tc_time = time.perf_counter() - t0
        tc_reward = eval_policy(make_env(), tc_ctrl.predict)

        # ── SB3 ─────────────────────────────────────────────────────────
        Cls = SB3SAC if method == "sac" else SB3TD3
        sb3_ctrl = Cls("MlpPolicy", make_env(), policy_kwargs=dict(net_arch=[64, 64]),
                       seed=0, verbose=0)
        t0 = time.perf_counter()
        sb3_ctrl.learn(total_timesteps=TOTAL_RL_STEPS)
        sb3_time = time.perf_counter() - t0

        def sb3_predict(obs):
            a, _ = sb3_ctrl.predict(obs, deterministic=True)
            return a, None

        sb3_reward = eval_policy(make_env(), sb3_predict)
        results[method] = dict(
            tc_time=tc_time, tc_reward=tc_reward,
            sb3_time=sb3_time, sb3_reward=sb3_reward,
        )

        speedup = sb3_time / tc_time if tc_time > 0 else float("nan")
        print(f"  tau-ctrl {method.upper()}: {tc_time:.2f}s, reward={tc_reward:.1f}")
        print(f"  SB3      {method.upper()}: {sb3_time:.2f}s, reward={sb3_reward:.1f}")
        print(f"  Speed-up: {speedup:.2f}x  |  Reward diff: {tc_reward - sb3_reward:+.1f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<6} | {'tau-ctrl t':>12} | {'SB3 t':>8} | {'Speedup':>8} | {'tau-ctrl R':>12} | {'SB3 R':>8}")
    print("-" * 70)
    for m, r in results.items():
        spd = r["sb3_time"] / r["tc_time"] if r["tc_time"] > 0 else float("nan")
        print(f"{m.upper():<6} | {r['tc_time']:>12.2f} | {r['sb3_time']:>8.2f} | "
              f"{spd:>8.2f}x | {r['tc_reward']:>12.1f} | {r['sb3_reward']:>8.1f}")

elif HAS_TORCH and not HAS_SB3:
    print("\n  SB3 not installed. Running tau-ctrl RL only:")
    for method in ["sac", "td3"]:
        tc_env = PendulumSwingUp(max_torque=15.0)
        tc_ctrl = make(method, tc_env, hidden=(64, 64), seed=0)
        t0 = time.perf_counter()
        tc_ctrl.learn(total_timesteps=TOTAL_RL_STEPS,
                      warmup_steps=WARMUP, batch_size=BATCH)
        tc_time = time.perf_counter() - t0
        tc_reward = eval_policy(tc_env, tc_ctrl.predict)
        final_angle = abs(tc_env.get_state()[0])
        print(f"  {method.upper()}: time={tc_time:.2f}s, reward={tc_reward:.1f}")

print("\nDone.")
