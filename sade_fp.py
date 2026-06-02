import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import List, Union, Optional, Tuple

warnings.filterwarnings('ignore')

@dataclass
class _SaDEResult:
    x:       np.ndarray
    fun:     float
    nfev:    int
    ngen:    int
    history: List[float] = field(default_factory=list)

def _choose(n: int, k: int, excl: Union[int, list], rng) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    if isinstance(excl, int): excl = [excl]
    for e in excl: mask[e] = False
    pool = np.where(mask)[0]
    if len(pool) < k: return rng.choice(pool, size=k, replace=True)
    return rng.choice(pool, size=k, replace=False)

def _s1(pop, F, t, b, rng):
    r1, r2, r3 = _choose(len(pop), 3, t, rng)
    return pop[r1] + F * (pop[r2] - pop[r3])

def _s2(pop, F, t, b, rng):
    r1, r2, r3, r4 = _choose(len(pop), 4, [t, b], rng)
    return (pop[t] + F*(pop[b]-pop[t]) + F*(pop[r1]-pop[r2]) + F*(pop[r3]-pop[r4]))

def _s3(pop, F, t, b, rng):
    r1, r2, r3 = _choose(len(pop), 3, t, rng)
    return pop[t] + F*(pop[r1]-pop[t]) + F*(pop[r2]-pop[r3])

def _s4(pop, F, t, b, rng):
    r1, r2, r3, r4, r5 = _choose(len(pop), 5, t, rng)
    return pop[r1] + F*(pop[r2]-pop[r3]) + F*(pop[r4]-pop[r5])

_STRATS = [_s1, _s2, _s3, _s4]
_NS     = len(_STRATS)

def _repair(v, lb, ub):
    v      = np.round(v).astype(int)
    span   = np.maximum(ub - lb, 1)
    period = 2 * span
    y = np.mod(v - lb, period)
    y = np.where(y > span, period - y, y)
    return lb + y

def _cross(x, v, CR, rng):
    d = len(x)
    m = rng.random(d) < CR
    m[rng.integers(d)] = True
    return np.where(m, v, x)

class _SaDE:
    """Self-Adaptive Differential Evolution Engine"""
    def __init__(self, func, bounds, pop_size=50, max_evals=1000,
                 lp=10, patience=10, seed=None):
        self.func     = func
        self.dim      = len(bounds)
        self.lb       = np.array([b[0] for b in bounds], dtype=int)
        self.ub       = np.array([b[1] for b in bounds], dtype=int)
        self.N        = pop_size
        self.maxev    = max_evals
        self.lp       = lp
        self.patience = patience
        self.rng      = np.random.default_rng(seed)
        self.p        = np.ones(_NS) / _NS
        self.crm      = np.full(_NS, 0.5)
        self.ns       = [deque(maxlen=lp) for _ in range(_NS)]
        self.nf       = [deque(maxlen=lp) for _ in range(_NS)]
        self.crmem    = [deque(maxlen=lp) for _ in range(_NS)]

    def run_opt(self, init_pop=None):
        if init_pop is not None and init_pop.shape == (self.N, self.dim):
            pop = np.clip(np.round(init_pop).astype(int), self.lb, self.ub)
        else:
            pop = self.rng.integers(self.lb, self.ub + 1, size=(self.N, self.dim))

        fit        = np.array([self.func(pop[i]) for i in range(self.N)])
        nfev       = self.N
        bi         = int(np.argmin(fit))
        hist       = [float(fit[bi])]
        gen        = 0
        no_improve = 0
        best_ever  = float(fit[bi])

        while nfev < self.maxev:
            gen += 1
            for k in range(_NS):
                self.ns[k].append(0)
                self.nf[k].append(0)
                self.crmem[k].append([])
            for i in range(self.N):
                if nfev >= self.maxev: break
                k  = self.rng.choice(_NS, p=self.p)
                F  = float(np.clip(self.rng.normal(0.5, 0.3), 1e-6, 2.0))
                CR = float(np.clip(self.rng.normal(self.crm[k], 0.1), 0.0, 1.0))
                v  = _STRATS[k](pop, F, i, bi, self.rng).astype(float)
                u  = _repair(_cross(pop[i].astype(float), v, CR, self.rng),
                             self.lb, self.ub)
                fu = self.func(u);  nfev += 1
                if fu <= fit[i]:
                    pop[i], fit[i] = u, fu
                    self.ns[k][-1] += 1
                    self.crmem[k][-1].append(CR)
                    if fu < fit[bi]: bi = i
                else:
                    self.nf[k][-1] += 1

            if gen % self.lp == 0:
                self._upd_p()
                self._upd_crm()

            hist.append(float(fit[bi]))

            if self.patience > 0:
                if fit[bi] < best_ever - 1e-5:
                    best_ever = float(fit[bi]);  no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    break

        return _SaDEResult(x=pop[bi].copy(), fun=float(fit[bi]),
                           nfev=nfev, ngen=gen, history=hist)

    def _upd_p(self):
        ns = np.array([sum(q) for q in self.ns], dtype=float)
        nf = np.array([sum(q) for q in self.nf], dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(ns+nf > 0, ns/(ns+nf), 0.0)
        tot = r.sum()
        if tot > 0:
            self.p = 0.05 + 0.95*r/tot
            self.p /= self.p.sum()

    def _upd_crm(self):
        for k in range(_NS):
            flat = [cr for gen_list in self.crmem[k] for cr in gen_list]
            if flat:
                self.crm[k] = float(np.median(flat))


class SaDE_FP_CoxFitter:
    """
    Self-Adaptive Differential Evolution with Fractional-Polynomials (SaDE-FP) 
    for Continuous Covariates in Cox Proportional Hazards Models.
    
    Standalone fitting algorithm intended for downstream prediction and analysis.
    """
    POWER_SET = [None, -3, -2.5, -2, -1.5, -1, -0.5, -0.25,
                 0,    0.25, 0.5,  1,  1.5,  2,  2.5,  3]
    N_POWERS  = len(POWER_SET)

    def __init__(self, penalizer=0.0):
        self.penalizer = penalizer
        self.final_fp_model = None
        self.best_powers = []
        self.covariates = []
        self.strata_cols = []
        
        self._shifts = {}
        self._scales = {}
        self._train_means = {}
        
    def _canonical_key(self, indices):
        pairs = []
        for i in range(0, len(indices), 2):
            p1 = self.POWER_SET[indices[i]]
            p2 = self.POWER_SET[indices[i+1]]
            if p1 is None and p2 is None:
                pairs.append((None, None))
            elif p1 is None and p2 is not None:
                pairs.append((p2, None))
            elif p1 is not None and p2 is None:
                pairs.append((p1, None))
            else:
                if p1 <= p2: pairs.append((p1, p2))
                else: pairs.append((p2, p1))
        return tuple(pairs)

    def _generate_fp_features(self, features, powers, store_means=False):
        transformed = {}
        for col, (p1, p2) in zip(features, powers):
            active = sorted([p for p in (p1, p2) if p is not None])
            if not active: continue
            if len(active) == 1:
                p = active[0]
                arr = self._precomp.get((col, p))
                if arr is None: return None
                name = f'{col}_fp_{p}'
                m = float(arr.mean())
                transformed[name] = arr - m
                if store_means: self._train_means[name] = m
            else:
                pa, pb = active
                za = self._precomp.get((col, pa))
                if za is None: return None
                name1 = f'{col}_fp1_{pa}'
                m1 = float(za.mean())
                transformed[name1] = za - m1
                if store_means: self._train_means[name1] = m1
                if pa == pb:
                    zb_raw = za * self._log_x[col]
                    name2 = f'{col}_fp2_rep_{pb}'
                else:
                    zb_raw = self._precomp.get((col, pb))
                    if zb_raw is None: return None
                    name2 = f'{col}_fp2_{pb}'
                m2 = float(zb_raw.mean())
                transformed[name2] = zb_raw - m2
                if store_means: self._train_means[name2] = m2
        return transformed

    def _objective_function(self, x):
        key = self._canonical_key(x)
        if key in self.evaluation_cache:
            return self.evaluation_cache[key]

        powers = list(key)
        fp_cols = self._generate_fp_features(self.covariates, powers)
        if not fp_cols:
            self.evaluation_cache[key] = 1e10
            return 1e10

        df_model = pd.DataFrame({**self._const_df_dict, **fp_cols})

        try:
            cph = CoxPHFitter(penalizer=self.penalizer)
            cph.fit(df_model, duration_col=self.duration_col,
                    event_col=self.event_col, strata=self.strata_cols or None,
                    show_progress=False)
            k = 2 * len(cph.params_)
            pBIC = -2 * cph.log_likelihood_ + k * np.log(self._n_events)
            val = pBIC
        except Exception:
            val = 1e10

        self.evaluation_cache[key] = val
        if val < self.best_val: self.best_val = val
        return val

    def fit(self, df, covariates, duration_col, event_col, strata_cols=None,
            maxiter=60, pop_size=None, pop_multiplier=10, seed=42):
        
        self.covariates = covariates
        self.duration_col = duration_col
        self.event_col = event_col
        self.strata_cols = strata_cols or []
        
        self.df = df.copy()
        
        # Preprocessing: Shift to strictly positive & scale for numerical stability
        for col in self.covariates:
            x = self.df[col].to_numpy(dtype=float)
            if not np.all(np.isfinite(x)):
                raise ValueError(f"Covariate '{col}' contains non-finite (NaN/Inf) values.")
            xmin = float(x.min())
            shift = (float(np.ptp(x)) or 1.0) * 1e-3 - xmin if xmin <= 0 else 0.0
            x = x + shift
            rng_ptp = float(np.ptp(x))
            scale = 10.0 ** np.floor(np.log10(rng_ptp)) if rng_ptp > 0 else 1.0
            self._shifts[col] = shift
            self._scales[col] = scale
            self.df[col] = x / scale

        self._n_events = int(self.df[self.event_col].sum())
        
        # Precompute power transforms for fast access
        self._precomp = {}
        self._log_x = {}
        for col in self.covariates:
            x = self.df[col].values.astype(float)
            log_x = np.log(x)
            self._log_x[col] = log_x
            for p in [p for p in self.POWER_SET if p is not None]:
                z = log_x if p == 0 else np.power(x, p)
                if np.isfinite(z).all():
                    self._precomp[(col, p)] = z

        const_cols = {
            self.duration_col: self.df[self.duration_col].values,
            self.event_col:    self.df[self.event_col].values,
        }
        for c in self.strata_cols:
            const_cols[c] = self.df[c].values
            
        self._const_df = pd.DataFrame(const_cols, index=self.df.index)
        self._const_df_dict = self._const_df.to_dict('series')

        self.evaluation_cache = {}
        self.best_val = np.inf

        # Calculate population size dynamically
        dim = 2 * len(self.covariates)
        pop_min = max(20, 4 * dim)
        pop_max = max(min(15 * dim, max(pop_min, self._n_events)), pop_min)
        
        if pop_size is not None:
            NP = max(6, int(pop_size))
        else:
            NP = int(np.clip(int(round(pop_multiplier * dim)), pop_min, pop_max))

        max_evals = NP * maxiter
        lp = max(3, maxiter // 5)
        patience = max(3, maxiter // 2)

        rng = np.random.default_rng(seed)
        init_pop = rng.integers(0, self.N_POWERS, size=(NP, dim))
        
        # Run SaDE Engine
        engine = _SaDE(
            func=self._objective_function,
            bounds=[(0, self.N_POWERS-1)]*dim,
            pop_size=NP, max_evals=max_evals,
            lp=lp, patience=patience, seed=seed)
        result = engine.run_opt(init_pop=init_pop)
        
        best_indices = result.x
        self.best_powers = []
        for i in range(len(self.covariates)):
            p1 = self.POWER_SET[best_indices[2*i]]
            p2 = self.POWER_SET[best_indices[2*i+1]]
            if p1 is None and p2 is not None: p1, p2 = p2, None
            elif p1 is not None and p2 is not None and p1 > p2: p1, p2 = p2, p1
            self.best_powers.append((p1, p2))
            
        # Fit final model with the optimal powers
        fp_cols = self._generate_fp_features(self.covariates, self.best_powers, store_means=True)
        if fp_cols is None: fp_cols = {}
        
        self._df_fp_final = pd.DataFrame({**self._const_df_dict, **fp_cols})
        
        self.final_fp_model = CoxPHFitter(penalizer=self.penalizer)
        self.final_fp_model.fit(
            self._df_fp_final,
            duration_col=self.duration_col, event_col=self.event_col,
            strata=self.strata_cols or None, show_progress=False)
            
        return self

    def transform(self, df_new):
        """
        Transforms new data into the FP feature space (applies shifts, scales, and learned centering).
        """
        if self.final_fp_model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
            
        transformed_features = {}
        for col, (p1, p2) in zip(self.covariates, self.best_powers):
            shift = self._shifts[col]
            scale = self._scales[col]
            
            x = (df_new[col].values.astype(float) + shift) / scale
            log_x = np.log(x)
            active = sorted([p for p in (p1, p2) if p is not None])
            
            if not active: continue
            
            def xp(p): return log_x if p == 0 else np.power(x, p)
            
            if len(active) == 1:
                p = active[0]
                name = f'{col}_fp_{p}'
                transformed_features[name] = xp(p) - self._train_means[name]
            else:
                pa, pb = active
                name1 = f'{col}_fp1_{pa}'
                transformed_features[name1] = xp(pa) - self._train_means[name1]
                
                if pa == pb:
                    name2 = f'{col}_fp2_rep_{pb}'
                    transformed_features[name2] = (xp(pa) * log_x) - self._train_means[name2]
                else:
                    name2 = f'{col}_fp2_{pb}'
                    transformed_features[name2] = xp(pb) - self._train_means[name2]
                    
        return pd.DataFrame(transformed_features, index=df_new.index)

    def predict_partial_hazard(self, df_new):
        if self.final_fp_model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        X_fp = self.transform(df_new)
        return self.final_fp_model.predict_partial_hazard(X_fp)
        
    def predict_survival_function(self, df_new, times=None):
        if self.final_fp_model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        X_fp = self.transform(df_new)
        return self.final_fp_model.predict_survival_function(X_fp, times=times)

    def print_summary(self):
        if self.final_fp_model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        self.final_fp_model.print_summary()
