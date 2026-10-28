"""v3.6: v3.2 (beta-binomial emissions + calibrated gate) with RHO estimated from the data instead of supplied.

UNATTESTED research model.  Not SOTA.  Imports v3.2 (which patches v3 globally): use ONE variant per process.
rho is moment-estimated on CN-neutral SNVs from the right flank of the top VAF mode (not contaminated by lower
subclones), then rho_used = clip(rho_hat + Z * se, FLOOR, CEIL).  Env: PARTITION_Z (default 1.0), PARTITION_RHO_CEIL.
With fewer than GATE_MIN_NEUTRAL neutral SNVs the gate is off anyway; rho_used = FALLBACK for the EM emissions.
"""
import os
import numpy as np

import partition_model_v3_2 as bb
import partition_model_v3 as base

Z = float(os.environ.get('PARTITION_Z', '1.0'))
FLOOR = 0.001
CEIL = float(os.environ.get('PARTITION_RHO_CEIL', '0.03'))
FALLBACK = 0.005


def _half_sample_mode(x):
    x = np.sort(x)
    while len(x) > 3:
        h = (len(x) + 1) // 2
        w = x[h - 1:] - x[:len(x) - h + 1]
        i = int(np.argmin(w))
        x = x[i:i + h]
    return float(x.mean())


def estimate_rho(a, n):
    """(rho_hat, se) from alt/depth of CN-neutral SNVs, right flank of the top mode only."""
    v = a / n
    q = float(np.clip(_half_sample_mode(v[v >= np.median(v)]), 0.02, 0.5))
    flank = v > q
    af, nf = a[flank], n[flank]
    r = (af - nf * q) ** 2 / (nf * q * (1 - q)) - 1.0
    w = nf - 1.0
    rho = float((r * w).sum() / (w ** 2).sum())
    se = float(np.sqrt(((w ** 2) * (r - rho * w) ** 2).sum()) / (w ** 2).sum())
    return rho, se


def fit_predict(alt, depth, major, minor, purity, seed=0):
    a, n = np.asarray(alt, float), np.asarray(depth, float)
    neutral = (np.asarray(major) == 1) & (np.asarray(minor) == 1) & (n > 0)
    if neutral.sum() >= base.GATE_MIN_NEUTRAL:
        rho, se = estimate_rho(a[neutral], n[neutral])
        used = float(np.clip(rho + Z * se, FLOOR, CEIL))
    else:
        used = FALLBACK
    bb.RHO = used
    out = base.fit_predict(alt, depth, major, minor, purity, seed=seed)
    out['rho_used'] = used
    return out
