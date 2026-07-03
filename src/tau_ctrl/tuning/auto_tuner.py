"""High-level auto-tuner that orchestrates parameter optimization."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np

from .bayesian_opt import BayesianOptimizer
from .genetic_opt import GeneticOptimizer


class AutoTuner:
    """Automatically tune controller parameters against a cost function.

    Parameters
    ----------
    param_bounds : dict[str, tuple[float, float]]
        Bounds for each parameter, e.g. ``{"kp": (1, 500), "kd": (0.1, 50)}``.
    method : str
        Optimization method: "bayesian", "genetic", or "random".
    n_iterations : int
        Number of optimization iterations.
    n_episodes : int
        Number of episodes to average per parameter set.
    verbose : bool
    """

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]],
        method: str = "bayesian",
        n_iterations: int = 100,
        n_episodes: int = 5,
        verbose: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.param_bounds = param_bounds
        self.method = method
        self.n_iterations = n_iterations
        self.n_episodes = n_episodes
        self.verbose = verbose

        self._history: list[dict[str, Any]] = []

        if method == "bayesian":
            self._optimizer: BayesianOptimizer | GeneticOptimizer = BayesianOptimizer(
                param_bounds, seed=seed
            )
        elif method == "genetic":
            self._optimizer = GeneticOptimizer(param_bounds, seed=seed)
        else:
            raise ValueError(f"Unknown tuning method: {method}")

    def tune(
        self,
        cost_fn: Callable[[dict[str, float]], float],
        callback: Optional[Callable[[int, dict[str, float], float], None]] = None,
    ) -> dict[str, Any]:
        """Run the tuning loop.

        Parameters
        ----------
        cost_fn : callable
            Function that takes a param dict and returns a scalar cost (lower is better).
        callback : callable, optional
            Called after each iteration with (iteration, params, cost).

        Returns
        -------
        dict
            Best parameters and tuning history.
        """
        best_cost = float("inf")
        best_params: dict[str, float] = {}

        for i in range(self.n_iterations):
            t0 = time.perf_counter()
            params = self._optimizer.suggest()

            # Average cost over episodes
            costs = []
            for _ in range(self.n_episodes):
                costs.append(cost_fn(params))
            cost = float(np.mean(costs))

            self._optimizer.observe(params, cost)
            self._history.append({"iteration": i, "params": params.copy(), "cost": cost})

            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()

            if self.verbose:
                elapsed = time.perf_counter() - t0
                print(f"[iter {i:3d}] cost={cost:.4f}  best={best_cost:.4f}  ({elapsed:.1f}s)")

            if callback:
                callback(i, params, cost)

        return {
            "best_params": best_params,
            "best_cost": best_cost,
            "history": self._history,
            "method": self.method,
            "n_iterations": self.n_iterations,
        }

    @property
    def history(self) -> list[dict[str, Any]]:
        return self._history
