"""Single-sample bulk mutation partitioning v2: calibrated 1-vs-2 gate + binomial-mixture EM + dispersion-scaled penalized K.

UNATTESTED research model (simulation + SMC-Het development only).  Not SOTA.

    from partition_model_v2 import predict_clusters
    out = predict_clusters(alt, depth, phi=1.0, seed=0)   # out['labels'], out['k'], out['p_one_vs_two']

alt/depth: per-mutation alternate and total read counts of clean CN-neutral (1+1) SNVs; any N >= 60.
phi: Pearson overdispersion of a one-cluster fit for the assay (1.0 = binomial); trained per assay, never from the test tumor.
Selection: K=1 if a parametric-bootstrap best-split LRT (overdispersed null) has p > ALPHA; otherwise
K in 2..KMAX maximising loglik/phi - LAMBDA*(2K-1) of an EM binomial mixture initialised from an ordered DP.
"""
import numpy as np
from scipy.special import logsumexp

ALPHA = 0.02       # gate level, chosen on simulations (see partition_model.py)
LAMBDA = 20.0      # per-parameter penalty (nats), chosen on SMC-Het development tumors, plateau 12-40, LOTO-checked
KMAX = 6
BOOT = 300
MIN_SEGMENT = 10
CAP = 0.5          # max per-cluster VAF for a diploid CN-neutral SNV


def _ordered_dp(a, n, k, min_seg=8):
    o = np.argsort(a / n, kind='stable')
    ca = np.concatenate([[0], np.cumsum(a[o])]).astype(float)
    cn = np.concatenate([[0], np.cumsum(n[o])]).astype(float)
    m = len(a)
    idx = np.arange(m + 1)
    big_a = ca[None, :] - ca[:, None]
    big_n = cn[None, :] - cn[:, None]
    q = np.clip(np.minimum(CAP, big_a / np.maximum(big_n, 1)), 1e-9, 1 - 1e-9)
    ll = big_a * np.log(q) + (big_n - big_a) * np.log(1 - q)
    ll = np.where((idx[None, :] - idx[:, None]) >= min_seg, ll, -np.inf)
    best = np.full((k + 1, m + 1), -np.inf)
    arg = np.zeros((k + 1, m + 1), int)
    best[0, 0] = 0
    for j in range(1, k + 1):
        cand = best[j - 1][:, None] + ll
        arg[j] = cand.argmax(0)
        best[j] = cand.max(0)
    cuts = [m]
    for j in range(k, 0, -1):
        cuts.append(arg[j][cuts[-1]])
    cuts = cuts[::-1]
    labels = np.zeros(m, int)
    for s in range(k):
        labels[o[cuts[s]:cuts[s + 1]]] = s
    return labels


def _comp_ll(a, n, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return a[:, None] * np.log(q)[None] + (n - a)[:, None] * np.log(1 - q)[None]


def _em(a, n, init, k, iters=60):
    w = np.array([(init == j).mean() for j in range(k)]).clip(1e-3)
    q = np.array([min(CAP, a[init == j].sum() / max(1, n[init == j].sum())) for j in range(k)])
    for _ in range(iters):
        lp = _comp_ll(a, n, q) + np.log(w)[None]
        r = np.exp(lp - logsumexp(lp, 1, keepdims=True))
        w = r.mean(0).clip(1e-4)
        w /= w.sum()
        q = np.minimum(CAP, (r * a[:, None]).sum(0) / np.maximum((r * n[:, None]).sum(0), 1))
    lp = _comp_ll(a, n, q) + np.log(w)[None]
    return lp.argmax(1), float(logsumexp(lp, 1).sum())


def _best_split_gain(a, n):
    o = np.argsort(a / n, kind='stable')
    a, n = a[o].astype(float), n[o].astype(float)
    ca, cn = np.cumsum(a), np.cumsum(n)

    def ll(x, y):
        q = np.clip(np.minimum(.4, x / y), 1e-9, 1 - 1e-9)
        return x * np.log(q) + (y - x) * np.log(1 - q)
    i = np.arange(MIN_SEGMENT - 1, len(a) - MIN_SEGMENT)
    two = ll(ca[i], cn[i]) + ll(ca[-1] - ca[i], cn[-1] - cn[i])
    return float(two.max() - ll(ca[-1], cn[-1]))


def _bootstrap_p(a, n, rho, seed):
    stat = _best_split_gain(a, n)
    q = float(np.clip(a.sum() / n.sum(), .01, .5))
    rng = np.random.default_rng([20261100, seed])
    exceed = 0
    for _ in range(BOOT):
        if rho > 1e-6:
            kap = 1. / rho - 1.
            p = rng.beta(q * kap, (1 - q) * kap, len(a))
        else:
            p = q
        if _best_split_gain(rng.binomial(n, p), n) >= stat:
            exceed += 1
    return (exceed + 1) / (BOOT + 1)


def predict_clusters(alt, depth, phi=1.0, seed=0):
    a = np.asarray(alt, float)
    n = np.asarray(depth, float)
    if len(a) != len(n) or len(a) < 60 or (n <= 0).any() or (a > n).any() or (a < 0).any():
        raise ValueError('need >=60 aligned mutations with 0 <= alt <= depth, depth > 0')
    if phi < 1.0:
        raise ValueError('phi must be >= 1')
    rho = max(0., (phi - 1.) / (n.mean() - 1.))
    p = _bootstrap_p(a.astype(int), n.astype(int), rho, seed)
    if p > ALPHA:
        return dict(labels=[0] * len(a), k=1, p_one_vs_two=p)
    fits = {}
    for k in range(2, KMAX + 1):
        lab, ll = _em(a, n, _ordered_dp(a, n, k), k)
        fits[k] = (lab, ll / phi - LAMBDA * (2 * k - 1))
    best = max(fits, key=lambda k: fits[k][1])
    _, relabeled = np.unique(fits[best][0], return_inverse=True)
    return dict(labels=relabeled.tolist(), k=int(relabeled.max() + 1), p_one_vs_two=p)
