"""Single-sample SNV partitioning v3: copy-number-aware binomial mixture in cellular-prevalence (CP) space.

UNATTESTED research model.  Not SOTA.

    from partition_model_v3 import fit_predict
    out = fit_predict(alt, depth, major, minor, purity, seed=0)
    # out['labels'] (0..K-1 per SNV), out['centers'] (CP of each cluster), out['k']

Per SNV i: expected VAF = m * c_k / D_i with D_i = purity*(major+minor) + 2*(1-purity), c_k = cluster CP (cell fraction),
and an unobserved integer multiplicity m in 1..max(major,1) (uniform prior).  Clusters are fit by EM on a fixed-size
subsample (K chosen by loglik - LAMBDA*(2K-1), the penalty calibrated at N=600), then re-fit with that K on up to NFIT SNVs;
every SNV is assigned by posterior argmax.  A parametric-bootstrap best-split LRT on CN-neutral SNVs may force K=1.
"""
import numpy as np
from scipy.special import logsumexp, gammaln

LAMBDA = 20.0
KMAX = 6
NSEL = 600
NFIT = 3000
GRID = 160
GATE_MIN_NEUTRAL = 150
ALPHA = 0.02
BOOT = 300
MIN_SEGMENT = 10


def _binom_logpmf(a, n, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return gammaln(n + 1) - gammaln(a + 1) - gammaln(n - a + 1) + a * np.log(q) + (n - a) * np.log1p(-q)


class _Data:
    def __init__(self, a, n, major, minor, purity):
        self.a = np.asarray(a, float)
        self.n = np.asarray(n, float)
        self.purity = float(purity)
        tot = np.asarray(major, float) + np.asarray(minor, float)
        self.D = np.maximum(purity * tot + 2.0 * (1.0 - purity), 0.2)
        self.M = np.clip(np.asarray(major, int), 1, 4)
        self.mmax = int(self.M.max())
        self.logprior = np.where(np.arange(1, self.mmax + 1)[None, :] <= self.M[:, None],
                                 -np.log(self.M)[:, None], -np.inf)
        self.neutral = (np.asarray(major) == 1) & (np.asarray(minor) == 1)

    def take(self, idx):
        d = _Data.__new__(_Data)
        d.a, d.n, d.D, d.M, d.logprior, d.neutral = (self.a[idx], self.n[idx], self.D[idx], self.M[idx],
                                                    self.logprior[idx], self.neutral[idx])
        d.purity, d.mmax = self.purity, self.mmax
        return d

    def comp_ll_m(self, c):
        """(N, K, M) log p(a_i, m | cluster k) up to the cluster weight."""
        ms = np.arange(1, self.mmax + 1)
        q = ms[None, None, :] * np.asarray(c)[None, :, None] / self.D[:, None, None]
        return _binom_logpmf(self.a[:, None, None], self.n[:, None, None], q) + self.logprior[:, None, :]

    def comp_ll(self, c):
        return logsumexp(self.comp_ll_m(c), axis=2)


def _init_centers(d, k):
    v = np.minimum(d.purity, (d.a / d.n) * d.D)           # CP assuming multiplicity 1
    o = np.argsort(v)
    m = len(v)
    idx = np.arange(m + 1)
    cs = np.concatenate([[0], np.cumsum(v[o])])
    sq = np.concatenate([[0], np.cumsum(v[o] ** 2)])
    cnt = (idx[None, :] - idx[:, None]).astype(float)
    sums = cs[None, :] - cs[:, None]
    ss = sq[None, :] - sq[:, None]
    cost = np.where(cnt >= 8, ss - sums ** 2 / np.maximum(cnt, 1), np.inf)
    best = np.full((k + 1, m + 1), np.inf)
    arg = np.zeros((k + 1, m + 1), int)
    best[0, 0] = 0
    for j in range(1, k + 1):
        cand = best[j - 1][:, None] + cost
        arg[j] = cand.argmin(0)
        best[j] = cand.min(0)
    cuts = [m]
    for j in range(k, 0, -1):
        cuts.append(arg[j][cuts[-1]])
    cuts = cuts[::-1]
    return np.array([v[o[cuts[s]:cuts[s + 1]]].mean() for s in range(k)])


def _em(d, k, iters=40):
    grid = np.linspace(0.01, d.purity, GRID)
    c = np.clip(_init_centers(d, k), 0.02, d.purity)
    w = np.full(k, 1.0 / k)
    ms = np.arange(1, d.mmax + 1)
    q = ms[None, None, :] * grid[None, :, None] / d.D[:, None, None]           # (N, G, M)
    grid_ll = _binom_logpmf(d.a[:, None, None], d.n[:, None, None], q)
    for _ in range(iters):
        llm = d.comp_ll_m(c) + np.log(w)[None, :, None]
        tot = logsumexp(llm.reshape(len(d.a), -1), axis=1)
        r = np.exp(llm - tot[:, None, None])                                   # (N, K, M)
        w = np.clip(r.sum(2).mean(0), 1e-4, None)
        w /= w.sum()
        new_c = np.array([grid[int(np.argmax((r[:, j, None, :] * grid_ll).sum((0, 2))))] for j in range(k)])
        done = np.max(np.abs(new_c - c)) < 1e-4
        c = new_c
        if done:
            break
    llm = d.comp_ll_m(c) + np.log(w)[None, :, None]
    return c, w, float(logsumexp(llm.reshape(len(d.a), -1), axis=1).sum())


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


def _gate_p(a, n, seed):
    stat = _best_split_gain(a, n)
    q = float(np.clip(a.sum() / n.sum(), .01, .5))
    rng = np.random.default_rng([20261500, seed])
    exceed = sum(_best_split_gain(rng.binomial(n.astype(int), q), n) >= stat for _ in range(BOOT))
    return (exceed + 1) / (BOOT + 1)


def fit_predict(alt, depth, major, minor, purity, seed=0):
    a = np.asarray(alt, float)
    n = np.asarray(depth, float)
    keep = n > 0
    full = _Data(a[keep], n[keep], np.asarray(major)[keep], np.asarray(minor)[keep], purity)
    total = len(full.a)
    rng = np.random.default_rng([20261501, seed])
    sel = full.take(np.sort(rng.choice(total, min(NSEL, total), replace=False)))
    scores = {k: _em(sel, k)[2] - LAMBDA * (2 * k - 1) for k in range(1, KMAX + 1)}
    k_best = max(scores, key=scores.get)
    gate_p = None
    neu = np.where(full.neutral)[0]
    if len(neu) >= GATE_MIN_NEUTRAL:
        sub = rng.choice(neu, min(len(neu), 600), replace=False)
        gate_p = _gate_p(full.a[sub], full.n[sub], seed)
        if gate_p > ALPHA:
            k_best = 1
        elif k_best == 1:
            k_best = max((kk for kk in scores if kk >= 2), key=scores.get)
    c, w, _ = _em(full.take(np.sort(rng.choice(total, min(NFIT, total), replace=False))), k_best)
    post_ll = full.comp_ll(c) + np.log(w)[None, :]
    lab = post_ll.argmax(1)
    used, lab = np.unique(lab, return_inverse=True)
    post = np.exp(post_ll - logsumexp(post_ll, axis=1)[:, None])[:, used]
    post /= post.sum(1, keepdims=True)
    labels = np.zeros(len(a), int)
    labels[keep] = lab
    post_full = np.zeros((len(a), post.shape[1]))
    post_full[keep] = post
    post_full[~keep, 0] = 1.0
    return dict(post=post_full, labels=labels.tolist(), centers=c[used].tolist(), k=int(len(used)), gate_p=gate_p,
                scores={int(k): float(v) for k, v in scores.items()})
