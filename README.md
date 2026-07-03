# tau-ctrl

**SB3-like, simulator-agnostic control algorithms — feedback, sampling-based MPC, safety filtering, and GPU-native RL behind one interface.**

Like Stable-Baselines3, but for the whole controller spectrum, over any `gymnasium.Env`. No simulator dependency: works with whatever hands you an env (e.g. [tau-sim](https://tau-intelligence.com/tau_sim/)). Unlike SB3, the RL methods train **on-device** and scale to vectorized environments for real GPU speedup.

## Installation

```bash
pip install tau-ctrl            # PID, MPPI/CEM/iCEM, iLQR, CBF (numpy + scipy + gymnasium)
pip install tau-ctrl[torch]     # + RL: PPO, SAC, TD3, and vectorized on-device training
```

## Usage

```python
from tau_ctrl import make

ctrl = make("mppi", env, horizon=25, n_samples=300)   # or MPPI(env, ...), SAC(env), ...
action, _ = ctrl.predict(obs)                          # SB3-style
ctrl.learn(total_timesteps=100_000)                    # trainable methods (ppo/sac/td3)
ctrl.save("ctrl.pkl")
```

## Algorithms

| Method | Family | Needs | Notes |
|---|---|---|---|
| `pid` | feedback | obs only | independent PID/PD over selected obs indices |
| `mppi` | sampling MPC | `get_state`/`set_state` | Model-Predictive Path Integral; plans against the env's own reward; `noise_beta>0` for smoother ("colored-noise") torques |
| `cem` | sampling MPC | `get_state`/`set_state` | Cross-Entropy Method MPC |
| `icem` | sampling MPC | `get_state`/`set_state` | Improved CEM — colored noise + elite memory across iterations |
| `ilqr` | gradient MPC | `get_state`/`set_state` | Iterative LQR via finite-difference linearization; fast, precise convergence on smooth dynamics |
| `cbf` | safety filter | `get_state`/`set_state` | wraps any base controller, projects its action to keep `h(x) >= 0` |
| `ppo` | RL (on-policy) | torch | GPU-automatic via `device="auto"` |
| `sac` | RL (off-policy) | torch | replay buffer + twin critics + auto entropy tuning; far more sample-efficient than PPO |
| `td3` | RL (off-policy) | torch | replay buffer + twin critics + delayed policy updates + target smoothing |

Model-based methods (mppi/cem/icem/ilqr/cbf) need the env to be *branchable*
(expose `get_state()` / `set_state()`) so they can roll candidate sequences
forward without disturbing the live episode. All `torch`-based methods
auto-select `cuda` when available (`device="auto"`, including reproducible
seeding of torch's RNG).

## GPU-native RL: vectorized, on-device training

SB3 steps CPU environments and the policy update sits behind per-step Python.
tau-ctrl's SAC/TD3 instead run the replay buffer and the update **on the target
device**, and — given a batched env — step thousands of environments in parallel
with no numpy in the hot loop. `Trainer.auto` probes your env and hardware and
picks the fastest correct strategy, so the same call adapts across the whole
env-reality spectrum:

```python
import gymnasium as gym
from tau_ctrl import Trainer

# One line: probes env + hardware, wraps as needed, trains on the best engine.
model = Trainer.auto("sac", env=gym.make_vec("HalfCheetah-v4", num_envs=64),
                     total_timesteps=1_000_000)
action, _ = model.predict(obs)
```

| You have | Adapter | GPU helps |
|---|---|---|
| native `TorchVecEnv` (or MJX/Brax via `jax_to_torch`, Isaac Gym) | — (fits directly) | env **and** update on-device — the real win |
| `gymnasium.vector.VectorEnv` | `GymVectorAdapter` | buffer + update on device (env stepping stays CPU) |
| a single, non-batchable env (PyBullet, classic MuJoCo) + a factory | `SyncTorchVecEnv` | batched update on device |
| one env you can't replicate (a real robot) | — (single-env path) | only the update |

See [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) for head-to-head numbers vs
Stable-Baselines3 and skrl, and [`examples/`](examples/) for runnable scripts
(`quickstart.py`, `vectorized_rl.py`, `adaptive_training.py`).

## Safety filtering

```python
# CBF wraps any base controller and only intervenes when a barrier is at risk
from tau_ctrl import CBFFilter, make

base = make("pid", env, kp=8.0, kd=5.0, target=[0.0], q_idx=[0], dq_idx=[1])
safe = CBFFilter(env, base=base, barriers=lambda state: v_max - state[1], alpha=0.5)
action, _ = safe.predict(obs)
```

## Auto-tuning

```python
from tau_ctrl import AutoTuner

tuner = AutoTuner({"kp": (1, 500), "kd": (0.1, 50)}, method="bayesian", n_iterations=50)
result = tuner.tune(cost_fn)   # cost_fn: dict of params -> scalar cost
print(result["best_params"])
```

## Layout

```
src/tau_ctrl/
├── algorithms/   # base.py (interface + registry), off_policy.py (shared SAC/TD3 infra),
│   │             # pid.py, mppi.py (MPPI/CEM/ICEM), ilqr.py, cbf.py, ppo.py, sac.py, td3.py
│   │             # vec_env.py (TorchVecEnv), adapters.py, strategy.py (Trainer.auto)
│   └── envs/     # pure-Python toy envs for tests/examples
└── tuning/       # Bayesian & genetic auto-tuning
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Related

- [tau-sim](https://tau-intelligence.com/tau_sim/) — robotics environment builder
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — RL algorithms (API inspiration)
