"""Bayesian optimization for controller parameter tuning.

Uses Gaussian Process regression with Expected Improvement acquisition.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.stats import norm


class BayesianOptimizer:
    """Bayesian optimizer using Gaussian Process surrogate.

    Parameters
    ----------
    bounds : dict[str, tuple[float, float]]
    xi : float
        Exploration-exploitation trade-off for Expected Improvement.
    """

    def __init__(
        self, bounds: dict[str, tuple[float, float]], xi: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.n_dims = len(self.param_names)
        self.xi = xi
        self._rng = np.random.default_rng(seed)

        self._lower = np.array([b[0] for b in bounds.values()])
        self._upper = np.array([b[1] for b in bounds.values()])

        self._X: list[np.ndarray] = []
        self._y: list[float] = []

        # GP hyperparameters (simple RBF kernel)
        self._length_scale = 1.0
        self._signal_variance = 1.0
        self._noise_variance = 1e-6

    def suggest(self) -> dict[str, float]:
        """Propose the next set of parameters to evaluate."""
        if len(self._X) < 3:
            # Random samples for initial exploration
            x = self._rng.uniform(self._lower, self._upper)
        else:
            x = self._maximize_ei(num_candidates=1000)

        return {name: float(x[i]) for i, name in enumerate(self.param_names)}

    def observe(self, params: dict[str, float], cost: float) -> None:
        """Record an observation."""
        x = np.array([params[name] for name in self.param_names])
        self._X.append(x)
        self._y.append(cost)

        # Simple hyperparameter update (optional; could use MLE)
        if len(self._X) > 5:
            X_arr = np.array(self._X)
            self._length_scale = max(np.std(X_arr, axis=0).mean(), 0.1)

    # ------------------------------------------------------------------
    # Internal: GP & EI
    # ------------------------------------------------------------------

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel."""
        sqdist = (
            np.sum(X1**2, axis=1).reshape(-1, 1)
            + np.sum(X2**2, axis=1)
            - 2 * X1 @ X2.T
        )
        return self._signal_variance * np.exp(-0.5 * sqdist / self._length_scale**2)

    def _gp_predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GP posterior mean and variance."""
        X = np.array(self._X)
        y = np.array(self._y)

        K = self._kernel(X, X) + self._noise_variance * np.eye(len(X))
        K_s = self._kernel(X, X_test)
        K_ss = self._kernel(X_test, X_test)

        K_inv = np.linalg.inv(K)
        mu = K_s.T @ K_inv @ y
        sigma2 = np.diag(K_ss) - np.sum((K_s.T @ K_inv) * K_s.T, axis=1)
        sigma2 = np.maximum(sigma2, 1e-10)
        return mu, sigma2

    def _expected_improvement(self, X_test: np.ndarray, y_best: float) -> np.ndarray:
        mu, sigma2 = self._gp_predict(X_test)
        sigma = np.sqrt(sigma2)
        improvement = y_best - mu - self.xi
        Z = improvement / np.maximum(sigma, 1e-10)
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-10] = 0.0
        return ei

    def _maximize_ei(self, num_candidates: int = 1000) -> np.ndarray:
        y_best = min(self._y)
        candidates = self._rng.uniform(self._lower, self._upper, size=(num_candidates, self.n_dims))
        ei = self._expected_improvement(candidates, y_best)
        return candidates[np.argmax(ei)]
