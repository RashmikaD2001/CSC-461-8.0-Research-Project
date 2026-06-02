# =============================================================================
#  FP-Cox Optimizer — Self-adaptive Differential Evolution (SaDE)
#  v2 — Bug Fixes + Optimisations
# =============================================================================
#
#  BUG FIXES (vs v1):
#  ==================
#  FIX 1 │ generate_fp_features — repeated-power detection was wrong
#         │   OLD: `if idx==1 and p==p1`
#         │        Triggered when the 2nd sorted power accidentally equalled p1
#         │        even for DISTINCT powers.
#         │        Example: powers=(2, 1) = sorted=[1, 2]
#         │        idx=1 = p=2, p==p1 (2==2) = True = incorrectly applies log!
#         │   FIX: sort active powers first, then `if pa == pb` (explicit check)
#
#  FIX 2 │ BIC denominator used n_events instead of n
#         │   OLD: `n_events = max(df[event_col].sum(), 2)` = log(n_events)
#         │        n_events < n = log(n_events) < log(n) = weaker penalty
#         │        = biased toward overly complex models
#         │   FIX: `n = len(df)` — standard BIC definition
#
#  FIX 3 │ Cache key did not account for swapped indices
#         │   OLD: (3, 11) and (11, 3) stored as two separate evaluations
#         │        but represent the same FP2(−2, 1) model = wasted evals
#         │   FIX: canonical_key() sorts each (p1_idx, p2_idx) pair before caching
#
#  FIX 4 │ No NaN / Inf guard after FP transformation
#         │   Large negative powers (−3) on values near 0 produce Inf/NaN
#         │   which silently crash CoxPHFitter
#         │   FIX: np.isfinite() check before fitting
#
#  OPTIMISATION IMPROVEMENTS:
#  ===========================
#  IMP 5 │ Combined objective: BIC − γ·n·concordance_index
#         │   gamma=0  = pure BIC (backward-compatible default)
#         │   gamma>0  = rewards predictive accuracy; multiplying by n keeps
#         │              BIC and c-index on the same scale
#
#  IMP 6 │ Warm-start population
#         │   Seeds generation-0 with known-good solutions (linear, sqrt, log,
#         │   common FP2 combinations) so SaDE does not waste early evaluations
#
#  IMP 7 │ init_pop parameter added to _SaDE.run()
#         │   Allows external warm-start arrays to be injected
#
#  IMP 8 │ Strategy probability floor raised 0.01 = 0.05
#         │   Faster recovery when a strategy gets unlucky early on
#
#  IMP 9 │ seed exposed as parameter to optimize()
#         │   Was hard-coded to 42 in v1; now user-controllable
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

warnings.filterwarnings("ignore")

# == Power set (index 0 = excluded / None) ====================================
POWER_SET = [None, -3, -2.5, -2, -1.5, -1, -0.5, -0.25,
              0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3]
N_POWERS  = len(POWER_SET)   # 16, valid indices: 0..15


# =============================================================================
#  SaDE engine  (IMP 7: init_pop added; IMP 8: strategy floor raised)
# =============================================================================

@dataclass
class _SaDEResult:
    x:       np.ndarray
    fun:     float
    nfev:    int
    ngen:    int
    history: List[float] = field(default_factory=list)


def _choose(n, k, excl, rng):
    cands = np.delete(np.arange(n), excl)
    return rng.choice(cands, size=k, replace=False)

def _s1(pop, F, t, b, rng):                          # DE/rand/1
    r1, r2, r3 = _choose(len(pop), 3, t, rng)
    return pop[r1] + F*(pop[r2] - pop[r3])

def _s2(pop, F, t, b, rng):                          # DE/rand-to-best/2
    r1, r2, r3, r4 = _choose(len(pop), 4, t, rng)
    return (pop[t] + F*(pop[b]  - pop[t])
                   + F*(pop[r1] - pop[r2])
                   + F*(pop[r3] - pop[r4]))

def _s3(pop, F, t, b, rng):                          # DE/current-to-rand/1
    r1, r2, r3 = _choose(len(pop), 3, t, rng)
    return pop[t] + F*(pop[r1] - pop[t]) + F*(pop[r2] - pop[r3])

def _s4(pop, F, t, b, rng):                          # DE/rand/2
    r1, r2, r3, r4, r5 = _choose(len(pop), 5, t, rng)
    return pop[r1] + F*(pop[r2] - pop[r3]) + F*(pop[r4] - pop[r5])

_STRATS = [_s1, _s2, _s3, _s4]
_NS     = len(_STRATS)


def _repair(v, lb, ub):
    """Round to integer then reflect out-of-bounds values back into [lb, ub]."""
    v = np.round(v).astype(int)
    lo = v < lb;  v[lo] = lb[lo] + (lb[lo] - v[lo])
    hi = v > ub;  v[hi] = ub[hi] - (v[hi] - ub[hi])
    return np.clip(v, lb, ub)


def _cross(x, v, CR, rng):
    d = len(x)
    m = rng.random(d) < CR
    m[rng.integers(d)] = True
    return np.where(m, v, x)


class _SaDE:
    def __init__(self, func, bounds, pop_size=50, max_evals=50_000,
                 lp=50, seed=None, callback=None):
        self.func = func;  self.bounds = bounds;  self.dim = len(bounds)
        self.lb   = np.array([b[0] for b in bounds], dtype=int)
        self.ub   = np.array([b[1] for b in bounds], dtype=int)
        self.N    = pop_size;  self.maxev = max_evals;  self.lp = lp
        self.cb   = callback
        self.rng  = np.random.default_rng(seed)
        self.p    = np.ones(_NS) / _NS
        self.crm  = np.full(_NS, 0.5)
        self.ns   = [deque() for _ in range(_NS)]
        self.nf   = [deque() for _ in range(_NS)]
        self.crok = [deque() for _ in range(_NS)]

    # IMP 7: accepts optional warm-start population
    def run(self, init_pop: Optional[np.ndarray] = None):
        if (init_pop is not None
                and init_pop.shape == (self.N, self.dim)):
            pop = np.clip(np.round(init_pop).astype(int), self.lb, self.ub)
        else:
            pop = np.column_stack([
                self.rng.integers(self.lb[d], self.ub[d]+1, self.N)
                for d in range(self.dim)
            ])

        fit  = np.array([self.func(pop[i]) for i in range(self.N)])
        nfev = self.N
        bi   = int(np.argmin(fit))
        hist = [float(fit[bi])]
        gen  = 0

        while nfev < self.maxev:
            gen += 1
            for i in range(self.N):
                if nfev >= self.maxev:
                    break
                k  = self.rng.choice(_NS, p=self.p)
                F  = float(np.clip(self.rng.standard_cauchy()*0.3 + 0.5, 1e-6, 2.0))
                CR = float(np.clip(self.rng.normal(self.crm[k], 0.1), 0.0, 1.0))
                v  = _STRATS[k](pop, F, i, bi, self.rng).astype(float)
                u  = _repair(_cross(pop[i].astype(float), v, CR, self.rng),
                              self.lb, self.ub)
                fu = self.func(u);  nfev += 1
                if fu <= fit[i]:
                    pop[i], fit[i] = u, fu
                    self.ns[k].append(1);  self.crok[k].append(CR)
                    if fu < fit[bi]:
                        bi = i
                else:
                    self.nf[k].append(1)

            if gen % self.lp == 0:
                self._upd_p();  self._upd_crm()

            hist.append(float(fit[bi]))
            if self.cb:
                self.cb(pop[bi], fit[bi], gen)

        return _SaDEResult(x=pop[bi].copy(), fun=float(fit[bi]),
                           nfev=nfev, ngen=gen, history=hist)

    def _upd_p(self):
        ns  = np.array([sum(q) for q in self.ns], dtype=float)
        nf  = np.array([sum(q) for q in self.nf], dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(ns + nf > 0, ns / (ns + nf), 0.0)
        tot = r.sum()
        if tot > 0:
            # IMP 8: floor raised 0.01 = 0.05 for faster strategy recovery
            self.p = 0.05 + 0.95 * r / tot
            self.p /= self.p.sum()
        for k in range(_NS):
            while len(self.ns[k]) > self.lp: self.ns[k].popleft()
            while len(self.nf[k]) > self.lp: self.nf[k].popleft()

    def _upd_crm(self):
        for k in range(_NS):
            if self.crok[k]:
                self.crm[k] = float(np.median(list(self.crok[k])))
            while len(self.crok[k]) > self.lp: self.crok[k].popleft()


# =============================================================================
#  FPCoxOptimizer  v2
# =============================================================================

class FPCoxOptimizer:
    """
    Selects optimal fractional polynomial powers for a Cox PH model
    using Self-adaptive Differential Evolution (SaDE).

    Parameters
    ----------
    df           : pandas DataFrame
    covariates   : list of continuous covariate column names
    duration_col : survival-time column name
    event_col    : event-indicator column name
    penalizer    : Cox ridge penalizer (default 0.01)
    gamma        : c-index weight in combined objective (default 0.0)

                   objective = BIC  −  gamma × n × concordance_index

                   gamma = 0   =  pure BIC minimisation (v1 behaviour)
                   gamma = 1   =  good starting point; balances BIC and
                                  c-index (multiply by n keeps both terms
                                  on the same numerical scale)
                   gamma > 1   =  increasingly prioritises c-index over
                                  parsimony; may overfit on small datasets
    """

    POWER_SET = [None, -3, -2.5, -2, -1.5, -1, -0.5, -0.25,
                  0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3]
    N_POWERS  = len(POWER_SET)   # 16

    def __init__(self, df, covariates, duration_col, event_col,
                 penalizer=0.01, gamma=0.0):
        self.covariates   = covariates
        self.duration_col = duration_col
        self.event_col    = event_col
        self.penalizer    = penalizer
        self.gamma        = gamma

        self.df = self._preprocess_positive(df, covariates)

        # Instance-level cache — no global state (FIX 3 also lives here)
        self.evaluation_cache: dict = {}
        self.best_val           = np.inf
        self.history: list      = []
        self.best_powers: list  = []
        self.final_fp_model     = None
        self.traditional_model  = None

    # == Preprocessing =========================================================
    @staticmethod
    def _preprocess_positive(df, features):
        """Shift features to be strictly positive (required for log/power)."""
        df = df.copy()
        for col in features:
            x = df[col].astype(float)
            df[col] = x - x.min() + 1e-5 if (x <= 0).any() else x
        return df

    # == FIX 3: canonical cache key ============================================
    @staticmethod
    def _canonical_key(indices):
        """
        Normalise each (p1_idx, p2_idx) pair so swapped indices share
        one cache entry.

        POWER_SET is monotone on non-None values, so sorting indices
        is equivalent to sorting power values:
          (11, 3)  =  (3, 11)   [same FP2(−2, 1) model]
          (11, 0)  =  (0, 11)   [same FP1(linear) model]
          ( 0, 0)  =  (0, 0)    [excluded — unchanged]
        """
        key = list(indices)
        for i in range(0, len(key), 2):
            a, b = key[i], key[i+1]
            if b == 0 and a != 0:          # move None (0) to front
                key[i], key[i+1] = 0, a
            elif a != 0 and b != 0:        # sort non-None pair ascending
                key[i], key[i+1] = min(a, b), max(a, b)
        return tuple(key)

    # == FIX 1: generate_fp_features (repeated-power bug fixed) ===============
    def _generate_fp_features(self, df, features, powers):
        """
        Apply FP transforms to all covariates.

        Convention:
          (None, None) = variable excluded
          (p,   None)  = FP1: x^p          (ln x when p == 0)
          (p1,  p2)    = FP2: x^p1, x^p2
                         repeated (p1==p2): second term = x^p · ln x

        FIX 1: old code used `if idx==1 and p==p1` to detect repeated powers.
        This incorrectly triggered on DISTINCT powers when p2 happened to equal
        p1 numerically after sorting (e.g. powers=(2,1) = sorted=[1,2],
        idx=1 gives p=2, p==p1=2 = True = log term applied to non-repeated FP).
        Fixed by comparing the two sorted active powers directly: `if pa == pb`.
        """
        transformed = {}
        for col, (p1, p2) in zip(features, powers):
            x      = df[col].values.astype(float)
            active = sorted([p for p in (p1, p2) if p is not None])

            if not active:                             # variable excluded
                continue

            if len(active) == 1:                       # FP1
                p = active[0]
                z = np.log(x) if p == 0 else np.power(x, p)
                transformed[f"{col}_fp_{p}"] = z

            else:                                      # FP2
                pa, pb = active[0], active[1]
                za = np.log(x) if pa == 0 else np.power(x, pa)
                transformed[f"{col}_fp1_{pa}"] = za

                if pa == pb:                           # repeated power: x^p · ln x
                    transformed[f"{col}_fp2_rep_{pb}"] = za * np.log(x)
                else:
                    zb = np.log(x) if pb == 0 else np.power(x, pb)
                    transformed[f"{col}_fp2_{pb}"] = zb

        return pd.DataFrame(transformed, index=df.index)

    # == FIX 2 + IMP 5: objective with correct BIC and optional c-index ========
    def _objective_function(self, x):
        """
        x : integer array of power indices (already integers from SaDE).

        FIX 2 — BIC now uses total n, not n_events.
          n_events < n  =  log(n_events) < log(n)  =  weaker complexity penalty
          =  v1 was biased toward selecting more parameters than warranted.
          Standard BIC definition: -2·logL + k·ln(n)  where n = total rows.

        IMP 5 — Combined objective (when gamma > 0):
          objective = BIC  −  gamma × n × concordance_index
          Multiplying c-index by n brings it to the same scale as BIC so
          gamma=1 is a natural neutral weight.

        FIX 3 — canonical_key() ensures symmetric index pairs share one entry.
        FIX 4 — NaN/Inf guard prevents silent crashes from large negative powers.
        """
        key = self._canonical_key(x)
        if key in self.evaluation_cache:
            return self.evaluation_cache[key]

        powers = [
            (self.POWER_SET[key[2*i]], self.POWER_SET[key[2*i+1]])
            for i in range(len(self.covariates))
        ]

        df_fp = self._generate_fp_features(self.df, self.covariates, powers)
        if df_fp.shape[1] == 0:                        # all variables excluded
            self.evaluation_cache[key] = 1e10
            return 1e10

        # FIX 4: guard against Inf/NaN (large negative powers near 0)
        if not np.isfinite(df_fp.values).all():
            self.evaluation_cache[key] = 1e10
            return 1e10

        df_model = df_fp.copy()
        df_model[self.duration_col] = self.df[self.duration_col].values
        df_model[self.event_col]    = self.df[self.event_col].values

        cph = CoxPHFitter(penalizer=self.penalizer)
        try:
            cph.fit(df_model,
                    duration_col=self.duration_col,
                    event_col=self.event_col,
                    show_progress=False)

            n   = len(self.df)                         # FIX 2: total n
            k   = len(cph.params_)
            bic = -2 * cph.log_likelihood_ + k * np.log(n)
            ci  = cph.concordance_index_               # lifelines computes this for free
            val = bic - self.gamma * n * ci            # IMP 5: combined objective

        except Exception:
            val = 1e10

        self.evaluation_cache[key] = val
        if val < self.best_val:
            self.best_val = val
        return val

    # == IMP 6: warm-start population ==========================================
    def _warm_start_pop(self, pop_size: int, rng: np.random.Generator) -> np.ndarray:
        """
        Seed generation 0 with sensible starting models so SaDE does not
        waste early evaluations on obviously bad solutions.

        Seeded rows (up to pop_size):
          0 : all FP1 linear       index 11 = power  1
          1 : all FP1 sqrt         index 10 = power  0.5
          2 : all FP1 log          index  8 = power  0  (= ln x)
          3 : all FP2(1, 0.5)      most common FP2 in clinical literature
          4 : all FP2(1, 2)        another common FP2
        Remaining rows: random.
        """
        dim  = 2 * len(self.covariates)
        pop  = rng.integers(0, self.N_POWERS, size=(pop_size, dim))

        NA, LIN, SQ, LG, Q = 0, 11, 10, 8, 13
        seeds = [
            np.tile([LIN, NA],  len(self.covariates)),
            np.tile([SQ,  NA],  len(self.covariates)),
            np.tile([LG,  NA],  len(self.covariates)),
            np.tile([LIN, SQ],  len(self.covariates)),
            np.tile([LIN, Q],   len(self.covariates)),
        ]
        for row, seed in enumerate(seeds[:min(len(seeds), pop_size)]):
            pop[row] = seed
        return pop

    # == SaDE generation callback ==============================================
    def _callback(self, best_x, best_f, gen):
        self.history.append(self.best_val)

    # == Main optimisation =====================================================
    def optimize(self, maxiter=50, popsize=15, seed=42):
        """
        Run SaDE to select optimal FP powers.

        Parameters
        ----------
        maxiter : maximum generations
        popsize : population size multiplier per dimension
                  (total pop = popsize × 2 × n_features)
        seed    : RNG seed  (IMP 9: was hard-coded to 42 in v1)
        """
        dim       = 2 * len(self.covariates)
        bounds    = [(0, self.N_POWERS - 1)] * dim
        pop_size  = popsize * dim
        max_evals = pop_size * maxiter

        print(f"Starting SaDE Optimisation  (gamma = {self.gamma})")
        print(f"  Features   : {len(self.covariates)}")
        print(f"  Population : {pop_size}  |  Generations : {maxiter}"
              f"  |  Budget : {max_evals} evals")
        print(f"  Objective  : "
              + (f"BIC − {self.gamma} × n × c-index  (combined)"
                 if self.gamma > 0 else "BIC only"))

        rng      = np.random.default_rng(seed)
        init_pop = self._warm_start_pop(pop_size, rng)   # IMP 6

        optimizer = _SaDE(
            func      = self._objective_function,
            bounds    = bounds,
            pop_size  = pop_size,
            max_evals = max_evals,
            lp        = 50,
            seed      = seed,
            callback  = self._callback,
        )
        result = optimizer.run(init_pop=init_pop)        # IMP 7

        best_indices = result.x
        print("\n--- Optimal Power Selection (SaDE) ---")
        for i in range(len(self.covariates)):
            p1 = self.POWER_SET[best_indices[2*i]]
            p2 = self.POWER_SET[best_indices[2*i+1]]
            self.best_powers.append((p1, p2))
            print(f"  {self.covariates[i]:<20}: p1 = {str(p1):<6}  p2 = {p2}")

        print(f"\n  Best objective : {result.fun:.4f}")
        print(f"  Generations    : {result.ngen}")
        print(f"  Evaluations    : {result.nfev}"
              f"  (cache hits: {result.nfev - len(self.evaluation_cache)})")

        self._plot_convergence()
        self._fit_final_models()

    # == Convergence plot ======================================================
    def _plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.history)+1), self.history,
                 marker='o', markersize=3, linestyle='-', color='steelblue',
                 label='Best objective per generation')
        plt.title('SaDE Convergence — FP Power Selection')
        plt.xlabel('Generation')
        plt.ylabel('BIC − γ·n·c-index' if self.gamma > 0 else 'BIC')
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # == Final model fit & comparison ==========================================
    def _fit_final_models(self):
        """Fit and compare FP Cox model vs traditional (linear) Cox model."""
        print("\n" + "="*65)
        print("FINAL MODEL COMPARISON")
        print("="*65)

        # Traditional Cox (all covariates, linear)
        df_trad = self.df[
            self.covariates + [self.duration_col, self.event_col]
        ].copy()
        self.traditional_model = CoxPHFitter(penalizer=self.penalizer)
        self.traditional_model.fit(df_trad,
                                   duration_col=self.duration_col,
                                   event_col=self.event_col,
                                   show_progress=False)

        # FP Cox (SaDE-selected powers)
        df_fp = self._generate_fp_features(
            self.df, self.covariates, self.best_powers)
        df_fp[self.duration_col] = self.df[self.duration_col].values
        df_fp[self.event_col]    = self.df[self.event_col].values
        self.final_fp_model = CoxPHFitter(penalizer=self.penalizer)
        self.final_fp_model.fit(df_fp,
                                duration_col=self.duration_col,
                                event_col=self.event_col,
                                show_progress=False)

        # Metrics (use FIX 2 BIC formula for both models)
        n = len(self.df)
        k_t  = len(self.traditional_model.params_)
        k_fp = len(self.final_fp_model.params_)
        bic_t  = -2*self.traditional_model.log_likelihood_ + k_t  * np.log(n)
        bic_fp = -2*self.final_fp_model.log_likelihood_    + k_fp * np.log(n)
        ci_t   = self.traditional_model.concordance_index_
        ci_fp  = self.final_fp_model.concordance_index_

        W = 22
        header = f"\n{'Metric':<25} | {'Traditional Cox':>{W}} | {'FP Cox (SaDE)':>{W}}"
        print(header)
        print("-" * len(header))
        print(f"{'Concordance index':<25} | {ci_t:>{W}.4f} | {ci_fp:>{W}.4f}")
        print(f"{'BIC':<25} | {bic_t:>{W}.2f} | {bic_fp:>{W}.2f}")
        print(f"{'Log-likelihood':<25} | {self.traditional_model.log_likelihood_:>{W}.4f}"
              f" | {self.final_fp_model.log_likelihood_:>{W}.4f}")
        print(f"{'Parameters (k)':<25} | {k_t:>{W}d} | {k_fp:>{W}d}")


# =============================================================================
#  Usage example
# =============================================================================
#
#  # Pure BIC optimisation (gamma=0 — backward-compatible with v1)
#  opt = FPCoxOptimizer(df=my_df, covariates=['age','bmi','bp'],
#                       duration_col='time', event_col='event')
#  opt.optimize(maxiter=50, popsize=15)
#
#  # Combined BIC + c-index (gamma=1 is a good starting point)
#  opt = FPCoxOptimizer(df=my_df, covariates=['age','bmi','bp'],
#                       duration_col='time', event_col='event',
#                       gamma=1.0)
#  opt.optimize(maxiter=50, popsize=15)
#
#  # Access results after optimisation
#  print(opt.best_powers)             # selected (p1, p2) per covariate
#  opt.final_fp_model.print_summary() # lifelines summary of the FP Cox fit
# =============================================================================
