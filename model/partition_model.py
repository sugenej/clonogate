"""Single-sample bulk mutation partitioning: DP hard fits + calibrated 1-vs-2 gate + beta-binomial CV above K=2.

UNATTESTED EXPLORATORY model (simulation only; not SOTA, not run through the governed pilot).

    from partition_model import predict_clusters
    out = predict_clusters(pub)     # out['labels'] (600 ints), out['k'], out['p_one_vs_two']

pub: dict(alt=[600 ints], depth=[600 ints], mutation_ids=[600 str], fold_ids=[600 ints in 0..2, 200 each],
          batch='assay0'|'assay1'|'assay2')
Needs the project venv and the recovery_lab3 folder (ordered-DP source) available at L3.
"""
import sys
import numpy as np
from scipy.special import logsumexp
from scipy.stats import betabinom, binom

L3 = 'C:/Users/sugen/Documents/Codex/2026-10-11/c-users-sugen-onedrive-documents-chatgpt/work/recovery_lab3'
if L3 not in sys.path:
    sys.path.append(L3)
import noise_ordered_binomial_partition_v1 as dp  # noqa: E402

# Per-assay Pearson dispersion of a one-cluster fit, trained on a separate fresh-seed cohort
# (20261022, 7001), single-regime cases only. New assays need their own estimate.
PHI = {'assay0': 1.005, 'assay1': 1.439, 'assay2': 3.389}
ALPHA = 0.02       # bootstrap significance level for 1 vs >=2 clusters (chosen on a fresh cohort)
Z_NEXT = 1.0       # paired CV z threshold for K=3,4 once K>=2 is established
BOOT = 300
MIN_SEGMENT = 10
ROWS = 600


def _fit(alt, depth, ids, k):
    return dp._solve(list(alt), list(depth), list(ids), k, dp._Meter(None), [])['models'][k]


def _labels_for(pub, k):
    model = _fit(pub['alt'], pub['depth'], pub['mutation_ids'], k)
    a = np.asarray(pub['alt'], float)
    n = np.asarray(pub['depth'], float)
    order = np.argsort(a / n, kind='stable')
    labels = np.zeros(ROWS, dtype=int)
    for j, seg in enumerate(model['segments']):
        labels[order[seg['begin']:seg['end']]] = j
    return labels.tolist()


def _loglik(a, n, centers, weights, rho):
    q = np.clip(np.asarray(centers) / 2., 1e-9, 1 - 1e-9)
    log_w = np.log(np.asarray(weights))
    if rho <= 1e-6:
        comp = np.stack([binom.logpmf(a, n, qc) for qc in q], 1)
    else:
        kap = 1. / rho - 1.
        comp = np.stack([betabinom.logpmf(a, n, qc * kap, (1 - qc) * kap) for qc in q], 1)
    return logsumexp(comp + log_w, axis=1)


def _best_split_gain(a, n):
    """Max over ordered 2-segment splits of the binomial log-likelihood gain over one segment."""
    o = np.argsort(a / n, kind='stable')
    a, n = a[o].astype(float), n[o].astype(float)
    cum_a, cum_n = np.cumsum(a), np.cumsum(n)
    tot_a, tot_n = cum_a[-1], cum_n[-1]

    def ll(x, y):
        q = np.clip(np.minimum(.4, x / y), 1e-9, 1 - 1e-9)
        return x * np.log(q) + (y - x) * np.log(1 - q)
    i = np.arange(MIN_SEGMENT - 1, ROWS - MIN_SEGMENT)
    two = ll(cum_a[i], cum_n[i]) + ll(tot_a - cum_a[i], tot_n - cum_n[i])
    return float(two.max() - ll(tot_a, tot_n))


def _bootstrap_p(a, n, rho, seed):
    stat = _best_split_gain(a, n)
    q = float(np.clip(a.sum() / n.sum(), .01, .5))
    rng = np.random.default_rng([20261100, seed])
    exceed = 0
    for _ in range(BOOT):
        if rho > 1e-6:
            kap = 1. / rho - 1.
            p = rng.beta(q * kap, (1 - q) * kap, ROWS)
        else:
            p = q
        if _best_split_gain(rng.binomial(n, p), n) >= stat:
            exceed += 1
    return (exceed + 1) / (BOOT + 1)


def _k_above_two(pub, rho):
    a, n = np.asarray(pub['alt']), np.asarray(pub['depth'])
    folds = np.asarray(pub['fold_ids'])
    scores = {k: np.zeros(ROWS) for k in (2, 3, 4)}
    for f in range(3):
        train, held = np.where(folds != f)[0], np.where(folds == f)[0]
        for k in (2, 3, 4):
            m = _fit(a[train].tolist(), n[train].tolist(), [pub['mutation_ids'][i] for i in train], k)
            scores[k][held] = _loglik(a[held], n[held], m['centers'], m['weights'], rho)
    current = 2
    for k in (3, 4):
        d = scores[k] - scores[current]
        se = d.std(ddof=1) / np.sqrt(ROWS)
        if se > 0 and d.mean() / se > Z_NEXT:
            current = k
    return current


def predict_clusters(pub, seed=0):
    """Return dict(labels, k, p_one_vs_two). seed makes the bootstrap p-value reproducible."""
    if pub.get('batch') not in PHI:
        raise ValueError('batch must be one of %s (dispersion trained only for these assays)' % sorted(PHI))
    if not all(len(pub[x]) == ROWS for x in ('alt', 'depth', 'mutation_ids', 'fold_ids')):
        raise ValueError('exactly %d aligned mutations required' % ROWS)
    a, n = np.asarray(pub['alt']), np.asarray(pub['depth'])
    rho = max(0., (PHI[pub['batch']] - 1.) / (n.mean() - 1.))
    p = _bootstrap_p(a, n, rho, seed)
    if p > ALPHA:
        return dict(labels=[0] * ROWS, k=1, p_one_vs_two=p)
    k = _k_above_two(pub, rho)
    return dict(labels=_labels_for(pub, k), k=k, p_one_vs_two=p)
