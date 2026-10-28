"""v3.4: v3 with greedy warm-start split/merge K selection (same penalty LAMBDA per extra cluster) instead of a K scan.

UNATTESTED research model.  Gate, data model, assignment are identical to v3.  Starting at K=1, repeatedly try splitting each
cluster (centres c*(1-d), c*(1+d)); accept the best split if loglik gain > LAMBDA*2.  Then try merging each pair; accept
if the loglik loss < LAMBDA*2.  Repeat until stable (max K=KMAX).
"""
import numpy as np

import partition_model_v3 as base
import partition_model_v3_1 as cont   # continuous M-step EM runner (_run); importing patches base._em

SPLIT_DELTA = 0.25
STEP_PENALTY = 2 * base.LAMBDA


def _fit(d, c0):
    return cont._run(d, len(c0), np.sort(c0), 40)


def _split_merge(d):
    c, w, ll = _fit(d, np.array([float(np.clip(np.average(np.minimum(d.purity, d.a / d.n * d.D)), 0.02, d.purity))]))
    for _ in range(12):
        changed = False
        best = None
        if len(c) < base.KMAX:
            for j in range(len(c)):
                cand = np.concatenate([np.delete(c, j), [c[j] * (1 - SPLIT_DELTA), min(d.purity, c[j] * (1 + SPLIT_DELTA))]])
                res = _fit(d, cand)
                if best is None or res[2] > best[2]:
                    best = res
            if best[2] - ll > STEP_PENALTY:
                c, w, ll = best
                changed = True
        if not changed and len(c) > 1:
            best = None
            order = np.argsort(c)
            cs, ws = c[order], w[order]
            for j in range(len(cs) - 1):
                merged = np.concatenate([np.delete(cs, [j, j + 1]),
                                         [(cs[j] * ws[j] + cs[j + 1] * ws[j + 1]) / (ws[j] + ws[j + 1])]])
                res = _fit(d, merged)
                if best is None or res[2] > best[2]:
                    best = res
            if ll - best[2] < STEP_PENALTY:
                c, w, ll = best
                changed = True
        if not changed:
            break
    return c, w, ll


def fit_predict(alt, depth, major, minor, purity, seed=0):
    a = np.asarray(alt, float)
    n = np.asarray(depth, float)
    keep = n > 0
    full = base._Data(a[keep], n[keep], np.asarray(major)[keep], np.asarray(minor)[keep], purity)
    total = len(full.a)
    rng = np.random.default_rng([20261501, seed])
    sel = full.take(np.sort(rng.choice(total, min(base.NSEL, total), replace=False)))
    c_sel, _, _ = _split_merge(sel)
    k_best = len(c_sel)
    gate_p = None
    neu = np.where(full.neutral)[0]
    if len(neu) >= base.GATE_MIN_NEUTRAL:
        sub = rng.choice(neu, min(len(neu), 600), replace=False)
        gate_p = base._gate_p(full.a[sub], full.n[sub], seed)
        if gate_p > base.ALPHA:
            k_best = 1
        elif k_best == 1:
            k_best = 2
    fit = full.take(np.sort(rng.choice(total, min(base.NFIT, total), replace=False)))
    if k_best == len(c_sel):
        c, w, _ = _fit(fit, c_sel)
    else:
        c, w, _ = cont._em(fit, k_best)
    lab = (full.comp_ll(c) + np.log(w)[None, :]).argmax(1)
    used, lab = np.unique(lab, return_inverse=True)
    labels = np.zeros(len(a), int)
    labels[keep] = lab
    return dict(labels=labels.tolist(), centers=c[used].tolist(), k=int(len(used)), gate_p=gate_p,
                scores={int(k_best): 0.0})
