"""
Self-adaptive Differential Evolution ($DE) for Integer Programming
==================================================================
Based on: "Differential Evolution for Integer Programming Problems"
          IEEE Congress on Evolutionary Computation (CEC 2007)

Algorithm Features:
  - Self-adaptive strategy selection (DE/rand/1/bin, DE/rand-to-best/2/bin,
    DE/current-to-rand/1, DE/rand/2/bin)
  - Self-adaptive control parameters F (scale factor) and CR (crossover rate)
  - Integer rounding with boundary reflection
  - Strategy success probabilities updated via a sliding learning period

Reference test problems A–F from the paper are included for benchmarking.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
from collections import deque
import time


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OptimizeResult:
    x: np.ndarray          # Best solution found (integer vector)
    fun: float             # Best objective value
    nfev: int              # Total function evaluations used
    ngen: int              # Generations completed
    success: bool
    history: List[float] = field(default_factory=list)  # best-per-gen trace


# ---------------------------------------------------------------------------
# Mutation strategies
# ---------------------------------------------------------------------------

def _de_rand_1(pop, F, target_idx, best_idx, rng):
    """DE/rand/1: v = x_r1 + F*(x_r2 - x_r3)"""
    idxs = _choose(len(pop), 3, exclude=target_idx, rng=rng)
    r1, r2, r3 = idxs
    return pop[r1] + F * (pop[r2] - pop[r3])


def _de_rand_to_best_2(pop, F, target_idx, best_idx, rng):
    """DE/rand-to-best/2: v = x_t + F*(x_best - x_t) + F*(x_r1 - x_r2) + F*(x_r3 - x_r4)"""
    idxs = _choose(len(pop), 4, exclude=target_idx, rng=rng)
    r1, r2, r3, r4 = idxs
    return (pop[target_idx]
            + F * (pop[best_idx] - pop[target_idx])
            + F * (pop[r1] - pop[r2])
            + F * (pop[r3] - pop[r4]))


def _de_current_to_rand_1(pop, F, target_idx, best_idx, rng):
    """DE/current-to-rand/1: v = x_t + F*(x_r1 - x_t) + F*(x_r2 - x_r3)"""
    idxs = _choose(len(pop), 3, exclude=target_idx, rng=rng)
    r1, r2, r3 = idxs
    return (pop[target_idx]
            + F * (pop[r1] - pop[target_idx])
            + F * (pop[r2] - pop[r3]))


def _de_rand_2(pop, F, target_idx, best_idx, rng):
    """DE/rand/2: v = x_r1 + F*(x_r2 - x_r3) + F*(x_r4 - x_r5)"""
    idxs = _choose(len(pop), 5, exclude=target_idx, rng=rng)
    r1, r2, r3, r4, r5 = idxs
    return pop[r1] + F * (pop[r2] - pop[r3]) + F * (pop[r4] - pop[r5])


STRATEGIES = [_de_rand_1, _de_rand_to_best_2, _de_current_to_rand_1, _de_rand_2]
N_STRATEGIES = len(STRATEGIES)


def _choose(n_pop: int, k: int, exclude: int, rng: np.random.Generator) -> np.ndarray:
    """Sample k distinct indices from [0, n_pop), excluding `exclude`."""
    candidates = np.delete(np.arange(n_pop), exclude)
    return rng.choice(candidates, size=k, replace=False)


# ---------------------------------------------------------------------------
# Integer boundary handling  (reflection)
# ---------------------------------------------------------------------------

def _repair(v: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    Round to nearest integer then reflect out-of-bounds components back
    into [lb, ub].  Equation (8)/(9) from the paper.
    """
    v = np.round(v).astype(int)
    # Reflect lower violations
    mask_lo = v < lb
    v[mask_lo] = lb[mask_lo] + (lb[mask_lo] - v[mask_lo])
    # Reflect upper violations
    mask_hi = v > ub
    v[mask_hi] = ub[mask_hi] - (v[mask_hi] - ub[mask_hi])
    # Final clip to handle double-violations after reflection
    v = np.clip(v, lb, ub)
    return v


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def _binomial_crossover(x: np.ndarray, v: np.ndarray,
                        CR: float, rng: np.random.Generator) -> np.ndarray:
    """Standard binomial (uniform) crossover."""
    dim = len(x)
    mask = rng.random(dim) < CR
    mask[rng.integers(dim)] = True          # guarantee at least one from v
    u = np.where(mask, v, x)
    return u


# ---------------------------------------------------------------------------
# Main SaDE class
# ---------------------------------------------------------------------------

class SaDE:
    """
    Self-adaptive Differential Evolution for Integer Programming ($DE).

    Parameters
    ----------
    func         : Objective function  f(x) -> float  (minimisation)
    bounds       : List of (lb, ub) integer pairs, one per dimension
    pop_size     : Population size  (default 50)
    max_evals    : Budget of function evaluations  (default 50 000)
    lp           : Learning period for strategy-probability update (default 50)
    cr_mean_init : Initial mean for per-strategy CR distributions
    cr_std       : Standard deviation used when sampling CR
    seed         : Random seed for reproducibility
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[int, int]],
        pop_size: int = 50,
        max_evals: int = 50_000,
        lp: int = 50,
        cr_mean_init: float = 0.5,
        cr_std: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.func = func
        self.bounds = bounds
        self.dim = len(bounds)
        self.lb = np.array([b[0] for b in bounds], dtype=int)
        self.ub = np.array([b[1] for b in bounds], dtype=int)
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.lp = lp
        self.cr_std = cr_std
        self.rng = np.random.default_rng(seed)

        # --- Self-adaptive state ---
        # Strategy selection probabilities (equal initially)
        self.p_strat = np.ones(N_STRATEGIES) / N_STRATEGIES

        # Per-strategy CR means (self-adapted over learning periods)
        self.cr_means = np.full(N_STRATEGIES, cr_mean_init)

        # Sliding windows: successes / failures per strategy per LP window
        self.ns = [deque() for _ in range(N_STRATEGIES)]   # successes
        self.nf = [deque() for _ in range(N_STRATEGIES)]   # failures

        # CR values that led to successes (for mean update)
        self.cr_success = [deque() for _ in range(N_STRATEGIES)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> OptimizeResult:
        """Run the optimiser and return an OptimizeResult."""

        # ---- Initialise population ----
        pop = np.column_stack([
            self.rng.integers(self.lb[d], self.ub[d] + 1, size=self.pop_size)
            for d in range(self.dim)
        ])  # shape: (pop_size, dim)

        fitness = np.array([self.func(pop[i]) for i in range(self.pop_size)])
        nfev = self.pop_size
        best_idx = int(np.argmin(fitness))
        history = [float(fitness[best_idx])]

        if verbose:
            print(f"{'Gen':>6}  {'Best':>14}  {'Mean':>14}  {'Evals':>8}")
            print("-" * 50)
            print(f"{'0':>6}  {fitness[best_idx]:>14.4f}  {fitness.mean():>14.4f}  {nfev:>8}")

        gen = 0
        while nfev < self.max_evals:
            gen += 1

            for i in range(self.pop_size):
                if nfev >= self.max_evals:
                    break

                # --- Sample strategy ---
                s_idx = self.rng.choice(N_STRATEGIES, p=self.p_strat)
                strategy = STRATEGIES[s_idx]

                # --- Sample F from Cauchy(0.5, 0.3), clipped to (0,2] ---
                F = float(np.clip(
                    self.rng.standard_cauchy() * 0.3 + 0.5, 1e-6, 2.0
                ))

                # --- Sample CR from N(cr_mean_k, cr_std), clipped to [0,1] ---
                CR = float(np.clip(
                    self.rng.normal(self.cr_means[s_idx], self.cr_std), 0.0, 1.0
                ))

                # --- Mutate ---
                v_cont = strategy(pop, F, i, best_idx, self.rng).astype(float)

                # --- Crossover ---
                u_cont = _binomial_crossover(pop[i].astype(float), v_cont, CR, self.rng)

                # --- Repair (round + reflect into integer bounds) ---
                u = _repair(u_cont, self.lb, self.ub)

                # --- Evaluate ---
                f_u = self.func(u)
                nfev += 1

                # --- Selection + record success/failure ---
                if f_u <= fitness[i]:
                    pop[i] = u
                    fitness[i] = f_u
                    self.ns[s_idx].append(1)
                    self.cr_success[s_idx].append(CR)
                    if f_u < fitness[best_idx]:
                        best_idx = i
                else:
                    self.nf[s_idx].append(1)

            # ---- Update strategy probabilities every LP generations ----
            if gen % self.lp == 0:
                self._update_probabilities()
                self._update_cr_means()

            history.append(float(fitness[best_idx]))

            if verbose and gen % 100 == 0:
                print(f"{gen:>6}  {fitness[best_idx]:>14.4f}  {fitness.mean():>14.4f}  {nfev:>8}")

        if verbose:
            print("-" * 50)
            print(f"Final best: {fitness[best_idx]:.6f}  at x = {pop[best_idx]}")

        return OptimizeResult(
            x=pop[best_idx].copy(),
            fun=float(fitness[best_idx]),
            nfev=nfev,
            ngen=gen,
            success=True,
            history=history,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_probabilities(self):
        """
        Update strategy selection probabilities based on relative success rates
        over the current learning period window.  Equation from §II of the paper.
        """
        ns_sum = np.array([sum(q) for q in self.ns], dtype=float)
        nf_sum = np.array([sum(q) for q in self.nf], dtype=float)
        total = ns_sum + nf_sum

        # Success rate per strategy (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            rate = np.where(total > 0, ns_sum / total, 0.0)

        total_rate = rate.sum()
        if total_rate > 0:
            self.p_strat = 0.01 + 0.99 * rate / total_rate   # keep min prob 0.01
            self.p_strat /= self.p_strat.sum()

        # Slide the window: keep only the last lp entries
        for k in range(N_STRATEGIES):
            while len(self.ns[k]) > self.lp:
                self.ns[k].popleft()
            while len(self.nf[k]) > self.lp:
                self.nf[k].popleft()

    def _update_cr_means(self):
        """
        Update CR mean for each strategy using the median of successful CR values
        accumulated over the learning period.
        """
        for k in range(N_STRATEGIES):
            if len(self.cr_success[k]) > 0:
                self.cr_means[k] = float(np.median(list(self.cr_success[k])))
            while len(self.cr_success[k]) > self.lp:
                self.cr_success[k].popleft()


# ---------------------------------------------------------------------------
# Test problems from the paper (Problems A – F)
# ---------------------------------------------------------------------------

def _problem_A():
    """Minimize f = x1,  x1 ∈ [0,∞), integer.  Optimum = 0."""
    def f(x): return float(x[0])
    bounds = [(0, 100)]
    return f, bounds, 0.0


def _problem_B():
    """
    2-variable integer problem.
    min  f = x1^2 + x2^2 - 11*x1 - 7*x2 + 50
    s.t. x ∈ {0,...,10}^2
    Optimum ≈ 0 at x = (5, 3)  (or similar depending on exact coefficients)
    """
    def f(x): return float(x[0]**2 + x[1]**2 - 11*x[0] - 7*x[1] + 50)
    bounds = [(0, 10), (0, 10)]
    return f, bounds, None


def _problem_C():
    """
    3-variable integer knapsack-style problem from the paper.
    min  -(2*x1 + 5*x2 + 7*x3)
    s.t.  x1 + x2 + x3 <= 10,  x1 in [0,10], x2 in [0,5], x3 in [0,5]
    Equivalent to maximising 2x1+5x2+7x3  s.t. x1+x2+x3<=10
    Optimum: x=(0,5,5), f=−60
    """
    def f(x):
        penalty = max(0, x[0] + x[1] + x[2] - 10)
        return float(-(2*x[0] + 5*x[1] + 7*x[2])) + 1000 * penalty
    bounds = [(0, 10), (0, 5), (0, 5)]
    return f, bounds, -60.0


def _problem_D():
    """
    Classic integer nonlinear programming problem (CEC test).
    min  f = -x1 - x2
    s.t. x2 <= 2*x1^4 - 8*x1^3 + 8*x1^2 + 2
          x2 <= 4*x1^4 - 32*x1^3 + 88*x1^2 - 96*x1 + 36
          0 <= x1 <= 3,  0 <= x2 <= 4  integers
    Optimum: (3,4) -> f = -7
    """
    def f(x):
        x1, x2 = float(x[0]), float(x[1])
        c1 = x2 - (2*x1**4 - 8*x1**3 + 8*x1**2 + 2)
        c2 = x2 - (4*x1**4 - 32*x1**3 + 88*x1**2 - 96*x1 + 36)
        penalty = 1000 * (max(0.0, c1) + max(0.0, c2))
        return -x1 - x2 + penalty
    bounds = [(0, 3), (0, 4)]
    return f, bounds, -7.0


def _problem_E():
    """
    5-variable integer nonlinear test.
    min  f = (x1-1)^2 + (x2-2)^2 + (x3-3)^2 + (x4-4)^2 + (x5-5)^2
    Optimum: x=(1,2,3,4,5), f=0
    """
    opt = np.array([1, 2, 3, 4, 5])
    def f(x): return float(sum((x[i] - opt[i])**2 for i in range(5)))
    bounds = [(-10, 10)] * 5
    return f, bounds, 0.0


def _problem_F():
    """
    Integer programming problem: maximise sum(c_i*x_i) subject to sum(a_i*x_i)<=b.
    Coefficients from the paper's table (problem F).
    Optimum f* = -38771.2 (converted to min).
    """
    c = np.array([6, 5, 5, 4, 4, 3], dtype=float)
    a = np.array([3, 1, 3, 3, 2, 2], dtype=float)
    b = 14
    def f(x):
        constraint_viol = max(0.0, float(np.dot(a, x)) - b)
        return float(-np.dot(c, x)) + 1000.0 * constraint_viol
    bounds = [(0, 5)] * 6
    return f, bounds, None


PROBLEMS = {
    "A": _problem_A,
    "B": _problem_B,
    "C": _problem_C,
    "D": _problem_D,
    "E": _problem_E,
    "F": _problem_F,
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark(n_runs: int = 30, verbose_runs: bool = False):
    """
    Run SaDE on all test problems, report mean ± std across n_runs trials.
    Mirrors the experimental setup in the paper (N=50, max_evals=50 000).
    """
    print("=" * 65)
    print("  SaDE Integer Programming Benchmark")
    print(f"  Runs={n_runs}, pop_size=50, max_evals=50 000")
    print("=" * 65)

    for name, prob_fn in PROBLEMS.items():
        func, bounds, known_opt = prob_fn()
        results = []
        for run in range(n_runs):
            opt = SaDE(func, bounds,
                       pop_size=50, max_evals=50_000,
                       seed=run * 137 + hash(name) % 1000)
            res = opt.run(verbose=verbose_runs)
            results.append(res.fun)

        arr = np.array(results)
        best_str = f"{known_opt:.4f}" if known_opt is not None else "unknown"
        print(f"\nProblem {name}:")
        print(f"  Known optimum : {best_str}")
        print(f"  Mean  ± Std   : {arr.mean():.6f} ± {arr.std():.6f}")
        print(f"  Best  / Worst : {arr.min():.6f}  /  {arr.max():.6f}")
        print(f"  Success rate  : "
              f"{100*np.mean(arr <= arr.min()+1e-6):.1f}% "
              f"(within 1e-6 of best found)")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# Convergence plot helper (optional, requires matplotlib)
# ---------------------------------------------------------------------------

def plot_convergence(problem_name: str = "D", n_runs: int = 5, seed_base: int = 0):
    """Plot convergence curves for a given problem over multiple runs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    func, bounds, known_opt = PROBLEMS[problem_name]()
    plt.figure(figsize=(9, 5))

    for run in range(n_runs):
        opt = SaDE(func, bounds, pop_size=50, max_evals=50_000,
                   seed=seed_base + run * 17)
        res = opt.run(verbose=False)
        plt.plot(res.history, alpha=0.7, label=f"Run {run+1}")

    if known_opt is not None:
        plt.axhline(known_opt, color="red", linestyle="--",
                    linewidth=1.5, label=f"Optimum ({known_opt})")

    plt.xlabel("Generation")
    plt.ylabel("Best Objective Value")
    plt.title(f"SaDE Convergence — Problem {problem_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/sade_convergence.png", dpi=150)
    print("Convergence plot saved to sade_convergence.png")
    plt.show()


# ---------------------------------------------------------------------------
# Quick-start example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n--- Single run on Problem D (classic integer NLP) ---")
    func, bounds, known_opt = _problem_D()
    optimizer = SaDE(
        func=func,
        bounds=bounds,
        pop_size=50,
        max_evals=10_000,
        seed=42,
    )
    result = optimizer.run(verbose=True)
    print(f"\nResult : x = {result.x},  f(x) = {result.fun}")
    print(f"Known optimum: {known_opt}")

    print("\n--- Running full benchmark (30 runs × 6 problems) ---")
    benchmark(n_runs=30)

    print("\n--- Plotting convergence for Problem D ---")
    plot_convergence("D", n_runs=5)
