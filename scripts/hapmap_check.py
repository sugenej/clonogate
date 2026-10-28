"""Real-read check of the detection law on the SMaHT HapMap6 mixture (germline variants with known mixture fractions).

Each truth SNV has a known expected VAF (the 'AF' of the benchmark VCF).  Sites are pooled by that VAF into 'clones', read
counts are thinned to a target depth, and ClonoGate (partition_v3, copy-neutral, purity 1) is run on pairs of classes (a
real two-cluster tumour with real read noise) and on single classes (a control: every split is false).  The detection law
(frozen) is then compared with the observed split rate.  Germline mixture, no copy-number change, no purity, no evolution.
Usage: hapmap_check.py counts | calibrate | run BUDGET_SECONDS | report
Inputs (scratchpad): hm/sites.tsv (chrom pos ref alt af) and hm/pileup.txt (samtools mpileup of the mixture CRAM).
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'model'))
SCRATCH = Path(__import__('os').environ.get('HAPMAP_DIR', 'data/hm'))
OUTDIR = HERE.parent / 'reports' / 'hapmap_check'
COUNTS, RUNS, REPORT = OUTDIR / 'counts.tsv.gz', OUTDIR / 'runs.jsonl', OUTDIR / 'HAPMAP_CHECK.md'
PAIRS = [(0.07, 0.05), (0.10, 0.05), (0.12, 0.05)]      # (clonal VAF, subclone VAF)
CONTROLS = [0.05, 0.10]
DEPTHS = [10, 15, 20, 30, 45, 0]                         # 0 = no thinning
N_PER_CLASS, REPS = 350, 30
B0, B1 = 2.36, 5.95                                      # frozen law (N = 700 and fs = 0.5, so x_eff = x)
MIN_DEPTH = 8
INDEL = re.compile(r'[+-](\d+)')


def strip_indels(bases: str) -> str:
    """Remove '+3ACG' / '-2TT' insertion and deletion annotations from an mpileup base string."""
    while (m := INDEL.search(bases)):
        bases = bases[:m.start()] + bases[m.end() + int(m.group(1)):]
    return bases


def parse_counts() -> None:
    sites = pd.read_csv(SCRATCH / 'sites.tsv', sep='\t', header=None, names=['chrom', 'pos', 'ref', 'alt', 'af'])
    alt_of = dict(zip(sites.pos, sites.alt))
    af_of = dict(zip(sites.pos, sites.af))
    rows = []
    for line in (SCRATCH / 'pileup.txt').read_text().splitlines():
        chrom, pos, ref, depth, bases = line.split('\t')[:5]
        pos = int(pos)
        if pos not in alt_of or int(depth) == 0:
            continue
        bases = strip_indels(re.sub(r'\^.', '', bases).replace('$', '')).upper()
        rows.append((pos, af_of[pos], int(depth), bases.count(alt_of[pos])))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=['pos', 'af', 'depth', 'alt']).to_csv(COUNTS, sep='\t', index=False)
    print(len(rows), 'sites')


def load() -> pd.DataFrame:
    d = pd.read_csv(COUNTS, sep='\t')
    med = d.depth.median()
    return d[(d.depth >= 0.5 * med) & (d.depth <= 2 * med)].reset_index(drop=True)


def bb_nll(params, a, n):
    q, rho = 1 / (1 + np.exp(-params[0])), 1 / (1 + np.exp(-params[1]))
    k = 1 / rho - 1
    return -np.sum(gammaln(n + 1) - gammaln(a + 1) - gammaln(n - a + 1) + betaln(a + q * k, n - a + (1 - q) * k) - betaln(q * k, (1 - q) * k))


def fit_rho(a, n) -> tuple:
    q0 = a.sum() / n.sum()
    res = minimize(bb_nll, [np.log(q0 / (1 - q0)), -5.0], args=(a, n), method='Nelder-Mead')
    return 1 / (1 + np.exp(-res.x[0])), 1 / (1 + np.exp(-res.x[1]))


def calibrate() -> dict:
    d = load()
    lines = ['## Calibration of the mixture CRAM at truth sites (chr20)', '',
             f'Median depth {d.depth.median():.0f}; sites kept within 0.5x-2x of the median. rho is the beta-binomial MLE.', '',
             '| truth VAF | sites | mean depth | observed VAF | rho (MLE) |', '|---|---|---|---|---|']
    rho = {}
    for af, g in d.groupby('af'):
        if len(g) < 200:
            continue
        _, r = fit_rho(g.alt.values.astype(float), g.depth.values.astype(float))
        rho[af] = r
        lines.append(f'| {af:g} | {len(g)} | {g.depth.mean():.1f} | {g.alt.sum() / g.depth.sum():.4f} | {r:.4f} |')
    print('\n'.join(lines))
    return {'lines': lines, 'rho': rho}


def thin(rng, alt, depth, depth_target, median):
    if depth_target == 0 or depth_target >= median:
        return alt, depth
    p = depth_target / median
    a = rng.binomial(alt, p)
    ref = rng.binomial(depth - alt, p)
    return a, a + ref


def draw(pool: pd.DataFrame, rng, n: int, depth_target: int, median: float):
    take = pool.sample(n, random_state=int(rng.integers(1 << 31)))
    return thin(rng, take.alt.values, take.depth.values, depth_target, median)


def run(budget: float) -> None:
    import partition_model_v3 as model
    import screen_compare as sc
    d = load()
    median = d.depth.median()
    pools = {af: g for af, g in d.groupby('af')}
    done = {json.loads(l)['key'] for l in RUNS.read_text().splitlines() if l.strip()} if RUNS.exists() else set()
    t0 = time.time()
    cells = [('pair', p) for p in PAIRS] + [('control', (c,)) for c in CONTROLS]
    for rep in range(REPS):
        for kind, spec in cells:
            for dt in DEPTHS:
                key = f'{kind}|{spec}|d{dt}|r{rep}'
                if key in done:
                    continue
                if time.time() - t0 > budget:
                    print('budget reached')
                    return
                rng = np.random.default_rng([8128, rep, dt, int(spec[0] * 1000), int(spec[-1] * 1000)])
                if kind == 'pair':
                    parts = [draw(pools[q], rng, N_PER_CLASS, dt, median) for q in spec]
                else:
                    parts = [draw(pools[spec[0]], rng, 2 * N_PER_CLASS, dt, median)]
                alt = np.concatenate([p[0] for p in parts])
                depth = np.concatenate([p[1] for p in parts])
                keep = depth >= MIN_DEPTH
                alt, depth = alt[keep], depth[keep]
                ones = np.ones(len(alt), int)
                res = model.fit_predict(alt, depth, ones, ones, 1.0, seed=rep)
                labels = np.asarray(res['labels'], int)
                cp = np.asarray(res['centers'], float)[labels]
                split = len(set(labels.tolist())) >= 2
                scores = sc.rule_scores(labels, cp, depth) if split else dict(gap=float('inf'), x_rho0=float('inf'), x_rho02=float('inf'))
                rec = dict(key=key, kind=kind, spec=list(spec), depth_target=dt, rep=rep, mean_depth=float(depth.mean()), n=int(len(alt)),
                           k=int(res['k']), split=bool(split), **{k: (None if np.isinf(v) else float(v)) for k, v in scores.items()})
                with RUNS.open('a') as f:
                    f.write(json.dumps(rec) + '\n')
                print(key, rec['k'], flush=True)
    print('ALL DONE')


def wilson(h, n):
    p, z = h / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - w, c + w


def kept(g) -> str:
    """Splits that survive the x >= 1 screen at an assumed rho of 0.02 (a missing score means the score was infinite)."""
    s = g[g.split]
    return f'{int((s.x_rho02.fillna(np.inf) >= 1).sum())}/{len(s)}'


def report() -> None:
    cal = calibrate()
    rho = cal['rho']
    runs = pd.DataFrame([json.loads(l) for l in RUNS.read_text().splitlines() if l.strip()])
    out = ['# Real-read check on the SMaHT HapMap6 mixture (auto-generated by scripts/hapmap_check.py)', '',
           'Germline variants with known mixture fractions (chr20, one Illumina CRAM); thinned to a target depth; copy-neutral, purity 1. '
           'Not a tumour: no copy-number change, no purity, no evolution.', ''] + cal['lines'] + ['']
    out += ['## Two-cluster pairs: observed split rate against the frozen law', '',
            '| clonal VAF | subclone VAF | target depth | mean depth | runs | split rate [95% CI] | x (rho measured) | law predicts | splits kept by x>=1 (rho 0.02) |',
            '|---|---|---|---|---|---|---|---|---|']
    for (q1, q2) in PAIRS:
        r = float(np.sqrt(rho.get(q1, 0.003) * rho.get(q2, 0.003)))
        for dt in DEPTHS:
            g = runs[(runs.kind == 'pair') & (runs.depth_target == dt) & runs.spec.apply(lambda s: s == [q1, q2])]
            if not len(g):
                continue
            md = g.mean_depth.mean()
            x = (q1 - q2) / np.sqrt(q1 * (1 - q1) * (1 / md + r))
            pred = 1 / (1 + np.exp(-(B0 + B1 * np.log(x))))
            lo, hi = wilson(int(g.split.sum()), len(g))
            out.append(f'| {q1} | {q2} | {dt or "full"} | {md:.0f} | {len(g)} | {g.split.mean():.2f} [{lo:.2f}, {hi:.2f}] | {x:.2f} | {pred:.2f} | {kept(g)} |')
    out += ['', '## Single-class controls (every split is false)', '',
            '| VAF | target depth | mean depth | runs | false-split rate [95% CI] | rho (MLE) | splits kept by x>=1 (rho 0.02) |', '|---|---|---|---|---|---|---|']
    for c in CONTROLS:
        for dt in DEPTHS:
            g = runs[(runs.kind == 'control') & (runs.depth_target == dt) & runs.spec.apply(lambda s: s == [c])]
            if not len(g):
                continue
            lo, hi = wilson(int(g.split.sum()), len(g))
            out.append(f'| {c} | {dt or "full"} | {g.mean_depth.mean():.0f} | {len(g)} | {g.split.mean():.2f} [{lo:.2f}, {hi:.2f}] | {rho.get(c, float("nan")):.4f} | {kept(g)} |')
    REPORT.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print('\n'.join(out))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'run':
        run(float(sys.argv[2]))
    else:
        {'counts': parse_counts, 'calibrate': calibrate, 'report': report}[cmd]()
