"""
Controlled survival simulation - Scenario B: NON-LINEAR Age effect.

The "different simulated example": same structure as Scenario A and the original
Kaggle dataset (single covariate Age, columns Age, Time, Event) but with a
genuinely non-linear, fractional-polynomial Age effect, so recovery of
non-linearity can be checked against a KNOWN truth.

Data-generating process (Cox PH, Weibull baseline; Bender, Augustin & Blettner,
Stat. Med. 2005):

    Age_i  ~ TruncNormal(mu=58, sd=12) on [25, 90]   # identical Age vector to Scenario A
    h(t|Age) = h0(t) * exp(eta(Age)),   h0(t) = lambda * nu * t^(nu-1)
    eta(Age) = beta * (Age / 58)^(-2)                # FP1 -> true power = -2
    T_i      = ( -log(U_i) / (lambda * exp(eta_i)) )^(1/nu),  U_i ~ Uniform(0,1)
    C_i      ~ Exponential(rate calibrated to TARGET_CENS)   # independent censoring
    Time_i   = min(T_i, C_i),   Event_i = 1{T_i <= C_i}

The effect is deliberately strong so the true power is the clearly isolated global
pBIC optimum at this sample size (SaDE recovers (-2, None) at maxiter=1500,
seed=42). eta is mean-centred so the baseline matches Scenario A.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ----- config -----------------------------------------------------------
OUTPUT_DIR  = Path('data/preprocess-data')
OUTPUT_FILE = 'preprocess_sim_nonlinear.csv'
N           = 200
SEED_AGE    = 11        # shared with generate_sim_linear.py -> identical Age vector
SEED_SURV   = 202
WEIB_SHAPE  = 1.4       # nu (same baseline as Scenario A)
WEIB_SCALE  = 0.03      # lambda
TRUE_POWER  = -2        # FP1 power on (Age / 58)
BETA        = 7.0       # strong effect so (-2, None) is the modal pBIC optimum at n=200
TARGET_CENS = 0.30


def sample_age(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(58, 12, size=int(n * 3))
    a = a[(a >= 25) & (a <= 90)]
    while len(a) < n:
        a = np.concatenate([a, rng.normal(58, 12, size=n)])
        a = a[(a >= 25) & (a <= 90)]
    return a[:n]


def calibrate_cens_rate(T, target, seed):
    rng = np.random.default_rng(seed)
    lo, hi = 1e-6, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        C = rng.exponential(1.0 / mid, size=len(T))
        if np.mean(T > C) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    age = sample_age(N, SEED_AGE)
    rng = np.random.default_rng(SEED_SURV)

    eta = BETA * (age / 58.0) ** TRUE_POWER
    eta = eta - eta.mean()

    U = rng.uniform(size=N)
    T = (-np.log(U) / (WEIB_SCALE * np.exp(eta))) ** (1.0 / WEIB_SHAPE)

    rate = calibrate_cens_rate(T, TARGET_CENS, SEED_SURV + 1)
    C = np.random.default_rng(SEED_SURV + 2).exponential(1.0 / rate, size=N)

    time  = np.minimum(T, C)
    event = (T <= C).astype(int)

    df = pd.DataFrame({'Age': np.round(age, 2),
                       'Time': np.round(time, 4),
                       'Event': event})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / OUTPUT_FILE
    df.to_csv(out, index=False)
    print(f'Saved {out}')
    print(f'  n={len(df)}  events={int(event.sum())}  '
          f'censoring={1 - event.mean():.1%}  median Time={np.median(time):.2f}')
    print(f'  True functional form: FP1 in Age, power = {TRUE_POWER}')


if __name__ == '__main__':
    main()
