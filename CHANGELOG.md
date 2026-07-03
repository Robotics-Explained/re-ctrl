# Changelog

All notable changes to `tau-ctrl` are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is
[SemVer](https://semver.org/).

## [Unreleased]

### Added
- **Vectorized, on-device RL** — `TorchVecEnv` interface + reference
  `TorchPendulum`; SAC/TD3 auto-detect a vectorized env and step it entirely on
  the target device (no numpy in the hot loop), with a batched
  `ReplayBuffer.add_batch`.
- **Env adapters** so the same trainer runs on any env reality:
  `GymVectorAdapter` (gymnasium.vector), `SyncTorchVecEnv` (N copies of a single
  non-batchable env), and `jax_to_torch` (zero-copy MJX/Brax via dlpack).
- **Adaptive spine** — `Trainer.auto(...)` probes env + hardware and dispatches
  to the fastest correct execution strategy; `probe_env` / `probe_hardware` /
  `select_strategy` exposed for transparency.
- Benchmarks under `benchmarks/` comparing tau-ctrl against other libraries (see `benchmarks/RESULTS.md`).

### Changed
- `gymnasium` is now a **core** dependency (the vectorized/adapter machinery is
  built on it and `import tau_ctrl` uses it directly).
- `requires-python` lowered to `>=3.9`.
- Full Apache-2.0 license text now shipped in `LICENSE`.

## [0.1.0]
- Initial layout: unified `BaseController` interface (`predict`/`learn`/
  `save`/`load`) over gymnasium envs; PID, MPPI/CEM/iCEM, iLQR, CBF, PPO, SAC,
  TD3; Bayesian/genetic auto-tuning.
