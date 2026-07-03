# Benchmarks

Dev-only comparison scripts (not shipped in the wheel). Results and analysis:
[`RESULTS.md`](RESULTS.md).

```bash
pip install tau-ctrl[benchmark]     # torch + stable-baselines3 + skrl

PYTHONPATH=src            python benchmarks/vs_sb3_benchmark.py   # tau-ctrl vs SB3
PYTHONPATH=src:benchmarks python benchmarks/skrl_benchmark.py     # skrl side
PYTHONPATH=src            python benchmarks/benchmark.py          # MPPI/CEM/iLQR + RL sweep
```

| File | What it does |
|---|---|
| `vs_sb3_benchmark.py` | tau-ctrl vs Stable-Baselines3 (single-env + the vectorized win) |
| `skrl_benchmark.py` | skrl SAC across the same envs (`_skrl_helpers.py` builds its models) |
| `_skrl_helpers.py` | skrl actor/critic/agent builders matched to the tau-ctrl setup |
| `benchmark.py` | broader sweep across the sampling-MPC planners + RL |

Run under a CUDA-enabled interpreter to get GPU numbers. Note the system `python`
here ships a `cu130` torch build that's too new for the installed driver — use
the conda env whose torch reports `cuda available: True`.
