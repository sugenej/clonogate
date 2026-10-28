"""Phylogeny over fitted clusters by minimum expected loss among sum-rule-feasible trees.  UNATTESTED; not SOTA.

    from partition_tree_v1 import infer_parents
    parents = infer_parents(centers, sizes, se)   # parents[k] = parent cluster index or -1 (normal) for each cluster k

Nodes are ordered by decreasing cellular prevalence; the largest-prevalence node is the root (parent normal).
Each other node takes a parent of strictly larger prevalence.  A tree is feasible if for every node the sum of its
children's prevalences <= its own prevalence + TOL_Z * sqrt(var terms) + TOL_ABS.  The returned tree maximises the
size-weighted pairwise similarity (ancestor/descendant, cousin and co-cluster relations) averaged over all feasible trees,
i.e. the Bayes decision under a uniform prior on feasible trees.
"""
import itertools
import numpy as np

TOL_Z = 2.5
TOL_ABS = 0.02


def _relations(parent, k):
    """parent: list over nodes 0..k-1 (parent index or -1). Returns (ancestor[i,j], cousin[i,j], same[i,j])."""
    anc = np.zeros((k, k), float)
    for j in range(k):
        p = parent[j]
        while p != -1:
            anc[p, j] = 1.0
            p = parent[p]
    same = np.eye(k)
    cousin = 1.0 - np.clip(anc + anc.T + same, 0, 1)
    return anc, cousin, same


def _wcorr(a, b, w):
    wa = (w * a).sum() / w.sum()
    wb = (w * b).sum() / w.sum()
    cov = (w * (a - wa) * (b - wb)).sum()
    va, vb = (w * (a - wa) ** 2).sum(), (w * (b - wb) ** 2).sum()
    return 1.0 if va < 1e-12 and vb < 1e-12 else (0.0 if va < 1e-12 or vb < 1e-12 else cov / np.sqrt(va * vb))


def _similarity(rel_a, rel_b, w):
    s = 0.0
    for x, y in zip(rel_a, rel_b):
        s += _wcorr(x.ravel(), y.ravel(), w.ravel())
    s += _wcorr(rel_a[0].T.ravel(), rel_b[0].T.ravel(), w.ravel())
    return s / 4.0


def _feasible(parent, c, se):
    k = len(c)
    for p in range(k):
        kids = [j for j in range(k) if parent[j] == p]
        if kids:
            tol = TOL_Z * np.sqrt(se[p] ** 2 + sum(se[j] ** 2 for j in kids)) + TOL_ABS
            if sum(c[j] for j in kids) > c[p] + tol:
                return False
    return True


def infer_parents(centers, sizes, se):
    c = np.asarray(centers, float)
    n = np.asarray(sizes, float)
    s = np.asarray(se, float)
    k = len(c)
    if k == 1:
        return [-1]
    order = np.argsort(-c, kind='stable')               # position -> original cluster
    co, no, so = c[order], n[order], s[order]
    cands = [[-1] + list(choice) for choice in itertools.product(*[range(j) for j in range(1, k)])]
    feas = [p for p in cands if _feasible(p, co, so)]
    if not feas:
        feas = cands
    w = np.outer(no, no)
    rels = [_relations(p, k) for p in feas]
    best, best_score = 0, -np.inf
    for i, ri in enumerate(rels):
        score = np.mean([_similarity(ri, rj, w) for rj in rels])
        if score > best_score + 1e-12:
            best, best_score = i, score
    chosen = feas[best]
    out = [-1] * k
    for pos in range(k):
        out[order[pos]] = -1 if chosen[pos] == -1 else int(order[chosen[pos]])
    return out


def chain_parents(centers):
    order = np.argsort(-np.asarray(centers, float), kind='stable')
    out = [-1] * len(order)
    for pos in range(1, len(order)):
        out[order[pos]] = int(order[pos - 1])
    return out


def star_parents(centers):
    order = np.argsort(-np.asarray(centers, float), kind='stable')
    out = [int(order[0])] * len(order)
    out[order[0]] = -1
    return out
