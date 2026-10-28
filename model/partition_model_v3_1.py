"""v3.1: v3 with a continuous bounded M-step (exact, concave) and multiple EM restarts.  UNATTESTED, not SOTA.

Only _em is replaced; gate, K selection, and assignment are identical to frozen v3.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

import partition_model_v3 as base

RESTARTS = 5
C_MIN = 0.01


def _mstep_c(d, rk, ms, hi):
    """argmax_c sum_{i,m} rk[i,m] * (a log q + (n-a) log(1-q)), q = m c / D_i; concave in c."""
    m = ms[None, :]
    D = d.D[:, None]
    a, n = d.a[:, None], d.n[:, None]

    def neg(c):
        q = np.clip(m * c / D, 1e-6, 1 - 1e-6)
        return -float((rk * (a * np.log(q) + (n - a) * np.log1p(-q))).sum())
    return float(minimize_scalar(neg, bounds=(C_MIN, hi), method='bounded', options={'xatol': 1e-5}).x)


def _run(d, k, c0, iters):
    ms = np.arange(1, d.mmax + 1)
    c = np.clip(np.asarray(c0, float), 0.02, d.purity)
    w = np.full(k, 1.0 / k)
    for _ in range(iters):
        llm = d.comp_ll_m(c) + np.log(w)[None, :, None]
        tot = logsumexp(llm.reshape(len(d.a), -1), axis=1)
        r = np.exp(llm - tot[:, None, None])
        w = np.clip(r.sum(2).mean(0), 1e-4, None)
        w /= w.sum()
        new_c = np.array([_mstep_c(d, r[:, j, :], ms, d.purity) for j in range(k)])
        done = np.max(np.abs(new_c - c)) < 1e-4
        c = new_c
        if done:
            break
    llm = d.comp_ll_m(c) + np.log(w)[None, :, None]
    return c, w, float(logsumexp(llm.reshape(len(d.a), -1), axis=1).sum())


def _em(d, k, iters=40):
    best = _run(d, k, base._init_centers(d, k), iters)
    if k > 1:
        rng = np.random.default_rng([20261502, k, len(d.a)])
        v = np.minimum(d.purity, (d.a / d.n) * d.D)
        for _ in range(RESTARTS - 1):
            cand = _run(d, k, np.sort(rng.choice(v, k, replace=False)), iters)
            if cand[2] > best[2]:
                best = cand
    return best


base._em = _em
fit_predict = base.fit_predict
