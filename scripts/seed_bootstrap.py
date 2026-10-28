"""Seed-cluster bootstrap for the detection-law results (no new simulations).

Every cell of each experiment was simulated with the same seeds (0..n-1), so cells are not independent and the cell bootstrap in
fit_boundary.py understates uncertainty.  Each result file keeps the per-seed cluster counts (`ks`), so here whole seeds are
resampled jointly across all cells: one bootstrap replicate draws seed ids with replacement and applies them to every cell, which
keeps the cross-cell dependence that shared seeds create.  Three results are re-interval-ed:
  1. the logistic boundary fit (x at 50% detection and the coefficients), next to the old cell bootstrap and a cell+seed bootstrap;
  2. RMSE / bias of the frozen law on the 48 new settings, and the folded-minus-simple RMSE difference;
  3. the pooled counts in the misspecified-rho table.
Seeds 0..29 are common to every cell; seeds 30..39 exist only in the 40-seed power_scale cells.
Writes reports/v3_falsesplit/SEED_BOOTSTRAP.md.  Usage: py -3 scripts/seed_bootstrap.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.append(str(Path(__file__).resolve().parent))
import fit_boundary as fb  # noqa: E402

RES = fb.RES
N_BOOT = 1000
SEED = 20261025
COMMON, TAIL = 30, 10  # seeds shared by every cell, and the extra seeds of the 40-seed cells
MISSPEC_SEEDS = 15


def draw_ids(rng: np.random.Generator) -> np.ndarray:
    """One replicate's seed ids for a 40-seed cell; a 30-seed cell uses the first 30 (the shared block)."""
    return np.concatenate([rng.integers(0, COMMON, COMMON), rng.integers(COMMON, COMMON + TAIL, TAIL)])


def resampled_hits(cells: list[dict], ids: np.ndarray) -> np.ndarray:
    """Detections per cell (K >= 2) after applying the same seed ids to every cell."""
    return np.array([float(np.sum(np.asarray(c['ks'])[ids[:c['n']]] >= 2)) for c in cells])


def interval(values: list[float] | np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def fit_summary(b: np.ndarray, with_cov: bool) -> dict:
    out = {'x50': fb.x50(b), 'b1': b[1]}
    if with_cov:
        out.update(a=b[2] / b[1], c=b[3] / b[1])
    return out


def boundary_replicates(sub: list[dict], with_cov: bool, rng: np.random.Generator) -> dict:
    """Refit under cell, seed, and cell+seed resampling; returns {scheme: [summary dict, ...]}."""
    X = fb.design(sub, with_cov)
    n = np.array([c['n'] for c in sub], float)
    out: dict[str, list[dict]] = {'cell': [], 'seed': [], 'cell+seed': []}
    for _ in range(N_BOOT):
        ids = draw_ids(rng)
        hits = resampled_hits(sub, ids)
        cell_idx = rng.integers(0, len(sub), len(sub))
        for scheme, idx, h in (('cell', cell_idx, np.array([c['hits'] for c in sub], float)),
                               ('seed', np.arange(len(sub)), hits), ('cell+seed', cell_idx, hits)):
            b = fb.fit(X[idx], h[idx], n[idx])
            if b[1] > 0:
                out[scheme].append(fit_summary(b, with_cov))
    return out


def boundary_section(name: str, sub: list[dict], with_cov: bool, rng: np.random.Generator) -> list[str]:
    X = fb.design(sub, with_cov)
    hits = np.array([c['hits'] for c in sub], float)
    n = np.array([c['n'] for c in sub], float)
    point = fit_summary(fb.fit(X, hits, n), with_cov)
    reps = boundary_replicates(sub, with_cov, rng)
    keys = ['x50', 'b1'] + (['a', 'c'] if with_cov else [])
    labels = {'x50': 'x at 50% detection', 'b1': 'slope on ln x', 'a': 'exponent on N/700', 'c': 'exponent on fs/0.5'}
    lines = [f'### {name} ({len(sub)} cells)', '', '| quantity | estimate | cell bootstrap (old) | seed bootstrap | cell + seed bootstrap |',
             '|---|---|---|---|---|']
    for k in keys:
        cols = [f'[{interval([r[k] for r in reps[s]])[0]:.2f}, {interval([r[k] for r in reps[s]])[1]:.2f}]'
                for s in ('cell', 'seed', 'cell+seed')]
        lines.append(f'| {labels[k]} | {point[k]:.2f} | ' + ' | '.join(cols) + ' |')
    kept = ', '.join(f'{s}: {len(reps[s])}/{N_BOOT}' for s in reps)
    return lines + ['', f'Replicates with an increasing curve kept: {kept}.', '']


def shared_seed_dependence(cells: list[dict]) -> list[str]:
    """Mean correlation of per-seed detection indicators between cells with intermediate detection (same seed block)."""
    mid = [c for c in cells if 0.2 <= c['hits'] / c['n'] <= 0.8 and c['n'] == COMMON + TAIL]
    ind = [np.asarray(c['ks'])[:COMMON] >= 2 for c in mid]
    corr = [np.corrcoef(a, b)[0, 1] for a, b in combinations(ind, 2) if a.std() > 0 and b.std() > 0]
    return [f'Shared-seed dependence: across {len(mid)} intermediate cells (detection 0.2-0.8) the mean pairwise correlation of the '
            f'per-seed detection indicators (seeds 0-29) is {np.mean(corr):+.3f} over {len(corr)} pairs; independent cells would give 0.', '']


def law_section(rng: np.random.Generator) -> list[str]:
    plan = json.loads((RES / 'law_validation_cells.json').read_text())
    runs = {r['key']: r for r in map(json.loads, (RES / 'law_validation.jsonl').read_text().splitlines())}
    cells = [{**c, **runs[c['key']]} for c in plan]
    assert all(len(c['ks']) == c['n'] == COMMON for c in cells)
    pred = {k: np.array([c[f'pred_{k}'] for c in cells]) for k in ('folded', 'simple')}
    gen = np.array([c['generator'] == 'bb' for c in cells])
    masks = {'All': np.ones(len(cells), bool), 'Fitted simulator': gen, 'Held-out generator': ~gen}
    stats: dict[str, list] = {f'{s}|{k}|{m}': [] for s in masks for k in pred for m in ('rmse', 'bias')}
    gap: list[float] = []
    for _ in range(N_BOOT):
        obs = resampled_hits(cells, draw_ids(rng)) / COMMON
        for s, mask in masks.items():
            for k, p in pred.items():
                err = (p - obs)[mask]
                stats[f'{s}|{k}|rmse'].append(float(np.sqrt(np.mean(err ** 2))))
                stats[f'{s}|{k}|bias'].append(float(err.mean()))
        gap.append(stats['All|simple|rmse'][-1] - stats['All|folded|rmse'][-1])
    obs0 = np.array([c['hits'] / c['n'] for c in cells])
    # Null: if the folded law were exactly right, 30 runs per setting alone would give this RMSE range (cells taken as independent).
    null_rmse = {s: [] for s in masks}
    for _ in range(N_BOOT):
        sim = rng.binomial(COMMON, pred['folded']) / COMMON
        for s, mask in masks.items():
            null_rmse[s].append(float(np.sqrt(np.mean((pred['folded'] - sim)[mask] ** 2))))
    lines = ['## 2. Frozen law on the 48 new settings (seed bootstrap over the 30 seeds)', '',
             'Folded RMSE is shown against the range that sampling noise alone would give if the law were exactly right '
             '(percentile interval of simulated observed rates, 30 runs per setting); a seed-bootstrap interval for a single RMSE '
             'is biased upward because resampling adds noise to rates that already contain it. Bias and the simple-form RMSE are '
             'shown with seed-bootstrap intervals.', '',
             '| settings | n | folded RMSE (noise-only range) | share of noise-only RMSEs >= observed | folded bias | simple RMSE | simple bias |',
             '|---|---|---|---|---|---|---|']
    for s, mask in masks.items():
        err = {k: (pred[k] - obs0)[mask] for k in pred}
        rmse = {k: float(np.sqrt(np.mean(err[k] ** 2))) for k in pred}
        nlo, nhi = interval(null_rmse[s])
        tail = float(np.mean(np.array(null_rmse[s]) >= rmse['folded']))
        row = [f'{rmse["folded"]:.3f} ({nlo:.3f}-{nhi:.3f})', f'{tail:.2f}']
        for k, m, fmt in (('folded', 'bias', '+'), ('simple', 'rmse', ''), ('simple', 'bias', '+')):
            val = float(err[k].mean()) if m == 'bias' else rmse[k]
            lo, hi = interval(stats[f'{s}|{k}|{m}'])
            row.append(f'{val:{fmt}.3f} [{lo:{fmt}.3f}, {hi:{fmt}.3f}]')
        lines.append(f'| {s} | {int(mask.sum())} | ' + ' | '.join(row) + ' |')
    glo, ghi = interval(gap)
    floor = float(np.sqrt(np.mean([c['pred_folded'] * (1 - c['pred_folded']) / c['n'] for c in cells])))
    return lines + ['', f'Simple-form RMSE minus folded-law RMSE (all 48): {np.mean(gap):.3f} [{glo:.3f}, {ghi:.3f}]; '
                    f'binomial noise floor (root of mean p(1-p)/30) at 30 runs per setting: {floor:.3f}.', '']


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def misspec_section(rng: np.random.Generator) -> list[str]:
    rows = [json.loads(l) for l in (RES / 'misspec_rho.jsonl').read_text().splitlines() if l.strip()]
    assert all(len(r['ks']) == r['n'] == MISSPEC_SEEDS for r in rows)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r['assumed'], r['delta']), []).append(r)
    reps: dict[tuple, list[int]] = {k: [] for k in groups}
    for _ in range(N_BOOT):
        ids = rng.integers(0, MISSPEC_SEEDS, MISSPEC_SEEDS)  # the same 15 tumours underlie every assumed rho and every cell
        for k, g in groups.items():
            reps[k].append(sum(int(np.sum(np.asarray(r['ks'])[ids] >= 2)) for r in g))
    deltas = sorted({k[1] for k in groups})
    lines = ['## 3. Misspecified rho: pooled counts with intervals (60 runs per entry, true rho 0.005)', '',
             'Each entry: count / 60, then the seed-bootstrap 95% interval over the 15 shared seeds and (in braces) the exact '
             'Clopper-Pearson interval that treats the 60 runs as independent. At 0/60 and 60/60 the seed bootstrap has no spread, '
             'so the Clopper-Pearson bound is the informative one there.', '',
             '| assumed rho | ' + ' | '.join(f'delta {d}' + (' (false splits)' if d == 0 else ' (detections)') for d in deltas) + ' |',
             '|---|' + '---|' * len(deltas)]
    for a in sorted({k[0] for k in groups}):
        cells = []
        for d in deltas:
            k, vals = int(sum(sum(1 for q in r['ks'] if q >= 2) for r in groups[(a, d)])), reps[(a, d)]
            slo, shi = interval(vals)
            clo, chi = clopper_pearson(k, 60)
            cells.append(f'{k}/60 [{slo / 60:.2f}, {shi / 60:.2f}] {{{clo:.2f}, {chi:.2f}}}')
        lines.append(f'| {a} | ' + ' | '.join(cells) + ' |')
    return lines + ['']


def main() -> None:
    rng = np.random.default_rng(SEED)
    cells = fb.load_cells()
    assert all(len(c['ks']) == c['n'] for c in cells)
    lines = ['# Seed-cluster bootstrap of the detection-law results (auto-generated by scripts/seed_bootstrap.py)', '',
             f'{N_BOOT} replicates, seed {SEED}. A replicate resamples whole seeds with replacement and applies them to every cell, '
             'so the dependence created by shared simulation seeds is kept. No new simulations were run.', '',
             '## 1. Detection boundary fit', '']
    main_grid = [c for c in cells if c['source'] == 'scale' and c['generator'] == 'bb' and c['n_snv'] == 700
                 and c['frac_sub'] == 0.5 and c['mult2'] == 0.0]
    pooled = [c for c in cells if c['generator'] == 'bb' and c['mult2'] == 0.0]
    lines += boundary_section('Main grid, x only', main_grid, False, rng)
    lines += boundary_section('Pooled bb cells with N and subclone-fraction terms', pooled, True, rng)
    lines += shared_seed_dependence(pooled)
    lines += law_section(rng) + misspec_section(rng)
    (RES / 'SEED_BOOTSTRAP.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
