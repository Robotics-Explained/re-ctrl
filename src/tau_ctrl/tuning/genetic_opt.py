"""Genetic algorithm optimization for controller parameter tuning."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class GeneticOptimizer:
    """Simple genetic algorithm optimizer.

    Parameters
    ----------
    bounds : dict[str, tuple[float, float]]
    population_size : int
    mutation_rate : float
    crossover_rate : float
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        population_size: int = 30,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_frac: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.n_dims = len(self.param_names)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_elite = max(1, int(elite_frac * population_size))
        self._rng = np.random.default_rng(seed)

        self._lower = np.array([b[0] for b in bounds.values()])
        self._upper = np.array([b[1] for b in bounds.values()])
        self._init_population()

        self._evaluated: list[tuple[np.ndarray, float]] = []
        self._idx = 0
        self._best_x: Optional[np.ndarray] = None
        self._best_cost = float("inf")

    def _init_population(self) -> None:
        self._population = self._rng.uniform(
            self._lower, self._upper, size=(self.population_size, self.n_dims)
        )

    def suggest(self) -> dict[str, float]:
        """Return next individual to evaluate (round-robin through population)."""
        if self._idx >= len(self._population):
            self._evolve()
            self._idx = 0
        x = self._population[self._idx]
        self._idx += 1
        return {name: float(x[i]) for i, name in enumerate(self.param_names)}

    def observe(self, params: dict[str, float], cost: float) -> None:
        x = np.array([params[name] for name in self.param_names])
        self._evaluated.append((x, cost))
        if cost < self._best_cost:
            self._best_cost = cost
            self._best_x = x.copy()

    def _evolve(self) -> None:
        """Create next generation via elitism, selection, crossover, mutation.

        Elitism carries the best individuals *ever seen* (not just this
        generation) forward unmutated, so a good solution found early can't
        be lost to genetic drift in a later generation. Mutation strength
        anneals with each generation so the population can explore broadly
        early on and fine-tune later.
        """
        if len(self._evaluated) < self.population_size:
            return  # Not enough evaluations yet

        self._n_generations = getattr(self, "_n_generations", 0) + 1
        anneal = 0.5 ** (self._n_generations - 1)  # halve mutation scale each gen

        # Build fitness (negative cost, higher is better)
        fitness = np.array([-c for _, c in self._evaluated])
        fitness = np.maximum(fitness - fitness.min() + 1e-6, 0)

        # Select parents
        parents_X = np.array([x for x, _ in self._evaluated])
        probs = fitness / fitness.sum()
        new_pop = np.zeros_like(self._population)

        # Elitism: always keep the best-ever individual(s), never mutated.
        assert self._best_x is not None
        elites = np.tile(self._best_x, (self.n_elite, 1))
        new_pop[: self.n_elite] = elites

        for i in range(self.n_elite, self.population_size, 2):
            p1 = parents_X[self._rng.choice(len(parents_X), p=probs)]
            p2 = parents_X[self._rng.choice(len(parents_X), p=probs)]

            if self._rng.random() < self.crossover_rate:
                alpha = self._rng.random(self.n_dims)
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation (annealed)
            for c in (c1, c2):
                mask = self._rng.random(self.n_dims) < self.mutation_rate
                c[mask] += self._rng.normal(0, anneal * 0.1 * (self._upper - self._lower))[mask]

            new_pop[i] = np.clip(c1, self._lower, self._upper)
            if i + 1 < self.population_size:
                new_pop[i + 1] = np.clip(c2, self._lower, self._upper)

        self._population = new_pop
        self._evaluated.clear()
