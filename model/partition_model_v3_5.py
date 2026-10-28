"""v3.5: v3 with data-selected overdispersion.  UNATTESTED research model.  Not SOTA.

Runs the v3.2 beta-binomial pipeline (emissions + gate null) on a grid of intra-class correlations RHO_GRID (0 = plain v3),
and keeps the run whose best penalised score max_k [loglik - LAMBDA*(2k-1)] minus RHO_PEN (for rho > 0) is largest.
The selected run's gate and fit are returned, so the false-split null matches the estimated noise level.
All runs share the same seed and therefore the same K-scan subsample, so scores are comparable.
"""
import partition_model_v3 as base

_ORIG_LL, _ORIG_GATE = base._binom_logpmf, base._gate_p
import partition_model_v3_2 as bb  # noqa: E402  (patches base to beta-binomial)

_BB_LL, _BB_GATE = base._binom_logpmf, base._gate_p
RHO_GRID = (0.0, 0.005, 0.01, 0.02, 0.04)
import os
RHO_PEN = float(os.environ.get("PARTITION_RHO_PEN", "10"))


def _run(rho, *args, **kwargs):
    if rho == 0.0:
        base._binom_logpmf, base._gate_p = _ORIG_LL, _ORIG_GATE
    else:
        bb.RHO = rho
        base._binom_logpmf, base._gate_p = _BB_LL, _BB_GATE
    try:
        return base.fit_predict(*args, **kwargs)
    finally:
        base._binom_logpmf, base._gate_p = _BB_LL, _BB_GATE


def fit_predict(alt, depth, major, minor, purity, seed=0):
    best, best_score = None, -float('inf')
    for rho in RHO_GRID:
        res = _run(rho, alt, depth, major, minor, purity, seed=seed)
        score = max(res['scores'].values()) - (RHO_PEN if rho > 0 else 0.0)
        if score > best_score:
            best, best_score = dict(res, rho=rho), score
    return best
