"""Tests for auto-tuner."""

import numpy as np
import pytest

from tau_ctrl.tuning import AutoTuner


def quadratic_cost(params: dict[str, float]) -> float:
    """Simple cost: (kp-50)^2 + (kd-5)^2."""
    return (params["kp"] - 50.0) ** 2 + (params["kd"] - 5.0) ** 2


class TestAutoTuner:
    def test_bayesian_optimization(self):
        bounds = {"kp": (1.0, 200.0), "kd": (0.1, 50.0)}
        tuner = AutoTuner(
            bounds, method="bayesian", n_iterations=20, n_episodes=1, verbose=False, seed=0
        )
        result = tuner.tune(quadratic_cost)

        assert "best_params" in result
        assert result["best_cost"] < 1.0  # should find near (50, 5)
        assert len(result["history"]) == 20

    def test_genetic_optimization(self):
        bounds = {"kp": (1.0, 200.0), "kd": (0.1, 50.0)}
        tuner = AutoTuner(
            bounds, method="genetic", n_iterations=90, n_episodes=1, verbose=False, seed=0
        )
        result = tuner.tune(quadratic_cost)

        assert result["best_cost"] < 5.0

    def test_callback(self):
        bounds = {"kp": (1.0, 200.0)}
        calls = []

        def cb(iteration, params, cost):
            calls.append((iteration, cost))

        tuner = AutoTuner(bounds, method="bayesian", n_iterations=10, n_episodes=1, verbose=False)
        tuner.tune(lambda p: p["kp"] ** 2, callback=cb)
        assert len(calls) == 10
