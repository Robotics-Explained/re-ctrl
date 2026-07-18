# Changelog

All notable changes to `tau-ctrl` are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is
[SemVer](https://semver.org/).

## [Unreleased]

### Added
- **`GRPO`** — a critic-free policy gradient for branchable envs. At each real
  step it branches `group_size` candidate actions from the *live* state
  (reusing the `get_state`/`set_state` contract MPPI/CEM/ILQR already use),
  scores them with group-relative (z-scored) returns instead of a learned
  value function, and updates with a PPO-style clipped surrogate. Benchmarked
  against PPO/SAC on Pendulum-v1 and HalfCheetah-v5 (dense reward) and a
  sparse/terminal-reward pendulum variant (3 seeds each, matched training
  budgets): GRPO consistently edges out PPO under sparse/terminal reward
  (mean −573 vs −588, low variance both) — the critic-free ranking sidesteps
  PPO's bootstrapping difficulty there — but **SAC's replay buffer still wins
  both the dense and sparse settings tested**. GRPO is offered for the
  branchable + sparse-reward niche, not as a general PPO/SAC replacement.

## [0.2.0]

### Added
- **`MujocoBranchable`** wrapper — makes any Gymnasium MuJoCo env (built-in or a
  custom `MujocoEnv` subclass) branchable for the model-based controllers by
  snapshotting/restoring MuJoCo's full physics state. `MPPI/CEM/ICEM/ILQR/CBF`
  now run on MuJoCo tasks: `make("mppi", MujocoBranchable(gym.make("HalfCheetah-v5")))`.

### Fixed
- **Model-based planners no longer corrupt or crash on wrapped `gym.make()` envs.**
  MPPI/CEM/ICEM/ILQR now roll candidate sequences out against the *unwrapped* env,
  so a `predict()` no longer advances `TimeLimit`'s counter (which silently
  truncated the live episode) — and ILQR no longer raises `IndexError` when a
  rollout would trip truncation.
- `tau_ctrl.__version__` now reflects the installed package (was hard-coded `0.1.0`).
- Clear errors instead of opaque failures: discrete-action envs
  (`TypeError: … requires a continuous Box action space`), a raw
  `gymnasium.vector.VectorEnv` passed to `make()` (points to `Trainer.auto`), and
  RL used without PyTorch (points to `pip install tau-ctrl[torch]`).

### Changed
- `get_state`/`set_state` now honor a branching contract provided anywhere on the
  wrapper stack (not only the unwrapped env), so branchable wrappers compose with
  `TimeLimit` and friends.

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
