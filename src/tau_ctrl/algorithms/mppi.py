"""Sampling-based model-predictive control: MPPI, CEM, and ICEM.

All plan by rolling *candidate action sequences* forward through the env and
scoring them with the env's **own reward** — so they are fully simulator- and
reward-agnostic. Rollouts branch from the current state via the env's
``get_state``/``set_state`` capability (see :mod:`.base`).

Noise: white by default (``noise_beta=0``, the original formulations); set
``noise_beta`` in (0, 1) for correlated ("colored") noise across the horizon,
which smooths the resulting action sequences ("Smooth-MPPI"). ``ICEM``
defaults to colored noise plus elite memory across iterations — see its
docstring.

Parallelism / GPU: rollouts are embarrassingly parallel. If you pass
``vector_env_fn`` (a callable returning a Gymnasium ``VectorEnv``), candidates
are evaluated as a batch — so a GPU-batched simulator env parallelises the
planner with no code change here. Otherwise a single env is stepped in a loop.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .base import BaseController, ControllerManifest, get_state, register, set_state


class _SamplingMPC(BaseController):
    """Shared machinery for MPPI/CEM: sample sequences, roll out, update."""

    manifest = ControllerManifest(
        name="sampling_mpc", family="sampling_mpc",
        requires_branching=True, uses_reward=True, gpu_capable=True,
    )

    def __init__(
        self,
        env: Any,
        horizon: int = 20,
        n_samples: int = 100,
        noise_sigma: float = 1.0,
        noise_beta: float = 0.0,
        gamma: float = 1.0,
        vector_env_fn: Optional[Callable[[int], Any]] = None,
        **kw: Any,
    ) -> None:
        self.horizon = int(horizon)
        self.n_samples = int(n_samples)
        self.noise_sigma = float(noise_sigma)
        self.noise_beta = float(noise_beta)
        self.gamma = float(gamma)
        self._vector_env_fn = vector_env_fn
        super().__init__(env, **kw)

    def _setup(self) -> None:
        self.nu = self.action_dim
        self._nominal = np.zeros((self.horizon, self.nu))
        # Roll candidate sequences out against the *unwrapped* env. get_state /
        # set_state already operate on env.unwrapped, so stepping the wrapped env
        # here would leak into stateful wrappers (TimeLimit._elapsed_steps,
        # OrderEnforcing) and corrupt/terminate the live episode — while the
        # dynamics and reward are identical either way.
        self._rollout_env = getattr(self.env, "unwrapped", self.env)
        self._venv = None
        if self._vector_env_fn is not None:
            self._venv = self._vector_env_fn(self.n_samples)

    def reset(self) -> None:
        self._nominal[:] = 0.0

    # --- noise -------------------------------------------------------------
    def _sample_noise(self, n_samples: int) -> np.ndarray:
        """Standard-normal noise, shape (n_samples, horizon, nu).

        ``noise_beta = 0`` (default) gives i.i.d. white noise per timestep —
        vanilla MPPI/CEM. ``noise_beta`` in (0, 1) generates *colored* noise
        via a first-order AR filter (unit variance preserved), correlating
        noise across the horizon so sampled action sequences — and the
        resulting torques — are smooth instead of jittery high-frequency
        chatter ("Smooth-MPPI", e.g. Vlahov et al.). Higher beta = smoother.
        """
        shape = (n_samples, self.horizon, self.nu)
        if self.noise_beta <= 0:
            return self.rng.normal(0.0, 1.0, shape)
        beta = self.noise_beta
        scale = np.sqrt(1.0 - beta ** 2)
        noise = np.empty(shape)
        noise[:, 0, :] = self.rng.normal(0.0, 1.0, (n_samples, self.nu))
        for t in range(1, self.horizon):
            noise[:, t, :] = beta * noise[:, t - 1, :] + scale * self.rng.normal(
                0.0, 1.0, (n_samples, self.nu)
            )
        return noise

    # --- rollouts --------------------------------------------------------
    def _returns(self, state0: Any, seqs: np.ndarray) -> np.ndarray:
        """Discounted return of each (K, H, nu) action sequence from ``state0``."""
        if self._venv is not None:
            return self._returns_vectorized(state0, seqs)
        return self._returns_serial(state0, seqs)

    def _returns_serial(self, state0: Any, seqs: np.ndarray) -> np.ndarray:
        K = seqs.shape[0]
        R = np.zeros(K)
        for k in range(K):
            set_state(self.env, state0)
            disc = 1.0
            for h in range(self.horizon):
                _, r, term, trunc, _ = self._rollout_env.step(seqs[k, h])
                R[k] += disc * r
                disc *= self.gamma
                if term or trunc:
                    break
        set_state(self.env, state0)  # leave the real env as we found it
        return R

    def _returns_vectorized(self, state0: Any, seqs: np.ndarray) -> np.ndarray:
        venv = self._venv
        venv.reset()
        # Broadcast the current state to every worker, if supported.
        if hasattr(venv, "set_state"):
            venv.set_state([state0] * self.n_samples)
        else:  # per-worker set via the standard call_* API
            venv.call("set_state", state0)
        R = np.zeros(self.n_samples)
        disc = 1.0
        done = np.zeros(self.n_samples, dtype=bool)
        for h in range(self.horizon):
            _, r, term, trunc, _ = venv.step(seqs[:, h])
            live = ~done
            R[live] += disc * np.asarray(r)[live]
            disc *= self.gamma
            done |= np.asarray(term) | np.asarray(trunc)
            if done.all():
                break
        return R

    # --- update rule (subclass) -----------------------------------------
    def _update(self, seqs: np.ndarray, returns: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, obs, state=None, deterministic: bool = True):
        state0 = get_state(self.env)
        noise = self._sample_noise(self.n_samples) * self.noise_sigma
        seqs = self._clip_seq(self._nominal[None] + noise)
        returns = self._returns(state0, seqs)
        set_state(self.env, state0)
        self._update(seqs, returns)

        action = self._clip_action(self._nominal[0])
        # Receding horizon: shift the plan forward one step.
        self._nominal = np.roll(self._nominal, -1, axis=0)
        self._nominal[-1] = 0.0
        return action, None

    def _clip_seq(self, seqs: np.ndarray) -> np.ndarray:
        sp = self.action_space
        if sp is not None and getattr(sp, "low", None) is not None:
            return np.clip(seqs, sp.low, sp.high)
        return seqs


@register("mppi")
class MPPI(_SamplingMPC):
    """Model-Predictive Path Integral control (softmax-weighted update)."""

    def __init__(self, env: Any, temperature: float = 1.0, **kw: Any) -> None:
        self.temperature = float(temperature)
        super().__init__(env, **kw)
        self.manifest = ControllerManifest(
            name="mppi", family="sampling_mpc",
            requires_branching=True, uses_reward=True, gpu_capable=True,
        )

    def _update(self, seqs: np.ndarray, returns: np.ndarray) -> None:
        adv = returns - returns.max()  # stabilise the softmax
        w = np.exp(adv / max(self.temperature, 1e-8))
        w = w / (w.sum() + 1e-12)
        self._nominal = np.einsum("k,khu->hu", w, seqs)


@register("cem")
class CEM(_SamplingMPC):
    """Cross-Entropy Method MPC (refit a Gaussian to the elite sequences)."""

    def __init__(self, env: Any, elite_frac: float = 0.1, iterations: int = 3, **kw: Any) -> None:
        self.elite_frac = float(elite_frac)
        self.iterations = int(iterations)
        super().__init__(env, **kw)
        self.manifest = ControllerManifest(
            name="cem", family="sampling_mpc",
            requires_branching=True, uses_reward=True, gpu_capable=True,
        )

    def predict(self, obs, state=None, deterministic: bool = True):
        # CEM refines the distribution over several iterations before acting.
        state0 = get_state(self.env)
        mean = self._nominal.copy()
        sigma = np.full((self.horizon, self.nu), self.noise_sigma)
        n_elite = max(1, int(self.elite_frac * self.n_samples))
        for _ in range(self.iterations):
            noise = self._sample_noise(self.n_samples)
            seqs = self._clip_seq(mean[None] + sigma[None] * noise)
            returns = self._returns(state0, seqs)
            set_state(self.env, state0)
            elite = seqs[np.argsort(returns)[-n_elite:]]
            mean = elite.mean(axis=0)
            sigma = elite.std(axis=0) + 1e-6
        self._nominal = mean
        action = self._clip_action(self._nominal[0])
        self._nominal = np.roll(self._nominal, -1, axis=0)
        self._nominal[-1] = 0.0
        return action, None

    def _update(self, seqs, returns):  # unused (predict is overridden)
        pass


@register("icem")
class ICEM(CEM):
    """Improved CEM (Pinneri et al. 2020): colored noise + elite memory.

    Two changes over vanilla CEM for the same sample budget:

    1. **Colored noise by default** — correlated across the horizon (see
       :meth:`_SamplingMPC._sample_noise`), giving smoother action sequences
       than i.i.d. Gaussian sampling.
    2. **Elite memory** — the single best sequence found so far *this
       planning step* is always injected back into the next iteration's
       population instead of being discarded when the distribution is
       refit. Vanilla CEM only keeps summary statistics (mean/std) of the
       elites, so a lucky early sample can be "forgotten"; here it can't be.
    """

    def __init__(self, env: Any, noise_beta: float = 0.2, **kw: Any) -> None:
        kw.setdefault("noise_beta", noise_beta)
        super().__init__(env, **kw)
        self.manifest = ControllerManifest(
            name="icem", family="sampling_mpc",
            requires_branching=True, uses_reward=True, gpu_capable=True,
        )

    def predict(self, obs, state=None, deterministic: bool = True):
        state0 = get_state(self.env)
        mean = self._nominal.copy()
        sigma = np.full((self.horizon, self.nu), self.noise_sigma)
        n_elite = max(1, int(self.elite_frac * self.n_samples))
        best_seq, best_return = None, -np.inf

        for _ in range(self.iterations):
            noise = self._sample_noise(self.n_samples)
            seqs = self._clip_seq(mean[None] + sigma[None] * noise)
            if best_seq is not None:
                seqs[0] = best_seq  # guarantee the best-so-far is re-evaluated & eligible
            returns = self._returns(state0, seqs)
            set_state(self.env, state0)

            order = np.argsort(returns)
            elite = seqs[order[-n_elite:]]
            if returns[order[-1]] > best_return:
                best_return = returns[order[-1]]
                best_seq = seqs[order[-1]].copy()
            mean = elite.mean(axis=0)
            sigma = elite.std(axis=0) + 1e-6

        self._nominal = best_seq if best_seq is not None else mean
        action = self._clip_action(self._nominal[0])
        self._nominal = np.roll(self._nominal, -1, axis=0)
        self._nominal[-1] = 0.0
        return action, None
