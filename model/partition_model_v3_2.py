"""v3.2: v3 with beta-binomial emissions (fixed intra-class correlation RHO) and a beta-binomial-calibrated gate.

UNATTESTED research model.  Not SOTA.  Set RHO before calling fit_predict (or env PARTITION_RHO).
Beta-binomial: a ~ BetaBin(n, alpha=q*kappa, beta=(1-q)*kappa), kappa = (1-RHO)/RHO.  The gate statistic is unchanged;
only its bootstrap null is replaced by the matching beta-binomial draw so false-split control is re-calibrated.
"""
import os
import numpy as np
from scipy.special import betaln, gammaln

import partition_model_v3 as base

RHO = float(os.environ.get('PARTITION_RHO', '0.01'))


def _kappa():
    return (1.0 - RHO) / RHO


def _bb_logpmf(a, n, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    k = _kappa()
    al, be = q * k, (1 - q) * k
    return (gammaln(n + 1) - gammaln(a + 1) - gammaln(n - a + 1)
            + betaln(a + al, n - a + be) - betaln(al, be))


def _gate_p(a, n, seed):
    stat = base._best_split_gain(a, n)
    q = float(np.clip(a.sum() / n.sum(), .01, .5))
    rng = np.random.default_rng([20261500, seed])
    k = _kappa()
    exceed = 0
    for _ in range(base.BOOT):
        p = rng.beta(q * k, (1 - q) * k, size=len(n))
        exceed += base._best_split_gain(rng.binomial(n.astype(int), p), n) >= stat
    return (exceed + 1) / (base.BOOT + 1)


base._binom_logpmf = _bb_logpmf
base._gate_p = _gate_p
fit_predict = base.fit_predict
