"""Real-read check of the detection law on computational mixtures of two Genome in a Bottle samples (HG002 + HG003, Illumina 60x).

Reads from the two real libraries are thinned and added at a chosen mixing fraction f of HG003; a site's expected VAF follows from
the two truth genotypes, VAF = ((1-f) g2 + f g3) / 2 (g = alt copies).  Sites with the same genotype pair form a 'clone'.  Two
VAF classes make a real two-cluster sample with real read noise (the standardised-gap setting at purity 1: shared hets at 0.5
against HG002-only hets at 0.5 (1-f)); one class is a control in which every split is false.  ClonoGate (partition_v3, copy-neutral,
purity 1) is run and compared with the frozen detection law and the x >= 1 screen.  Computational mixture of two libraries, not a
physical mixture and not a tumour: no copy-number change, no purity below 1, no clonal evolution.
Usage: giab_mixture_check.py sites | counts | run BUDGET_SECONDS | report
Inputs (scratchpad giab/): HG00[23]_GRCh38_1_22_v4.2.1_benchmark.vcf.gz and _noinconsistent.bed, pileup_HG002.txt, pileup_HG003.txt.
"""
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'model'))
from hapmap_check import fit_rho, strip_indels, wilson  # noqa: E402

GIAB = Path(os.environ.get('GIAB_DIR', 'data/giab'))
OUTDIR = HERE.parent / 'reports' / 'giab_mixture_check'
MODEL = os.environ.get('GIAB_MODEL', 'v3')               # 'v32' = beta-binomial model with the measured rho supplied (as in the frozen law test)
MODEL_RHO = float(os.environ.get('GIAB_RHO', '0.0025'))
SUFFIX = '' if MODEL == 'v3' else '_v32'
COUNTS, RUNS, REPORT = OUTDIR / 'counts.tsv.gz', OUTDIR / f'runs{SUFFIX}.jsonl', OUTDIR / f'GIAB_MIXTURE_CHECK{SUFFIX}.md'
CHROM, END = 'chr20', 30_000_000
# (name, clonal genotype pair, subclone genotype pair, mixing fractions of HG003)
PAIRS = [('A', (1, 1), (1, 0), (0.06, 0.10, 0.14, 0.20))]
CONTROLS = [(0.0, (1, 1)), (0.10, (1, 1))]
DEPTHS = [20, 30, 45, 60]
N_PER_CLASS, REPS, MIN_DEPTH = 350, 30, 8
B0, B1 = 2.36, 5.95                                      # frozen law (N = 700 and fs = 0.5, so x_eff = x)
WANTED = {(1, 1), (1, 0), (0, 1), (0, 2)}


def read_vcf(path: Path) -> dict:
    out = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            if not line.startswith(CHROM + '\t'):
                continue
            c = line.rstrip('\n').split('\t')
            pos = int(c[1])
            if pos > END:
                break
            if len(c[3]) == 1 and len(c[4]) == 1:
                out[pos] = (c[3], c[4], c[-1].split(':')[0].count('1'))
    return out


def read_bed(path: Path) -> tuple:
    iv = [(int(a[1]), int(a[2])) for a in (l.split('\t') for l in path.read_text().splitlines()) if a[0] == CHROM and int(a[1]) < END]
    iv.sort()
    return np.array([s for s, _ in iv]), np.array([e for _, e in iv])


def in_bed(bed: tuple, pos: int) -> bool:
    i = np.searchsorted(bed[0], pos - 1, side='right') - 1                  # BED is 0-based half-open
    return i >= 0 and pos - 1 < bed[1][i]


def make_sites() -> None:
    v2, v3 = (read_vcf(GIAB / f'HG00{n}_GRCh38_1_22_v4.2.1_benchmark.vcf.gz') for n in (2, 3))
    b2, b3 = (read_bed(GIAB / f'HG00{n}_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed') for n in (2, 3))
    rows = []
    for pos in sorted(set(v2) | set(v3)):
        ref, alt = (v2.get(pos) or v3[pos])[:2]
        if pos in v2 and pos in v3 and v2[pos][:2] != v3[pos][:2]:
            continue
        g2 = v2[pos][2] if pos in v2 else (0 if in_bed(b2, pos) else None)
        g3 = v3[pos][2] if pos in v3 else (0 if in_bed(b3, pos) else None)
        if g2 is not None and g3 is not None and (g2, g3) in WANTED:
            rows.append((pos, ref, alt, g2, g3))
    s = pd.DataFrame(rows, columns=['pos', 'ref', 'alt', 'g2', 'g3'])
    s.to_csv(GIAB / 'sites.tsv', sep='\t', index=False)
    pd.DataFrame({'c': CHROM, 's': s.pos - 1, 'e': s.pos}).to_csv(GIAB / 'sites.bed', sep='\t', index=False, header=False)
    print(len(s), 'sites;', s.groupby(['g2', 'g3']).size().to_dict())


def parse_pileup(path: Path, alt_of: dict) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        c = line.split('\t')
        pos = int(c[1])
        if pos in alt_of and int(c[3]) > 0:
            bases = strip_indels(re.sub(r'\^.', '', c[4]).replace('$', '')).upper()
            out[pos] = (int(c[3]), bases.count(alt_of[pos]))
    return out


def parse_counts() -> None:
    s = pd.read_csv(GIAB / 'sites.tsv', sep='\t')
    alt_of = dict(zip(s.pos, s.alt))
    p2, p3 = (parse_pileup(GIAB / f'pileup_HG00{n}.txt', alt_of) for n in (2, 3))
    keep = s[s.pos.isin(set(p2) & set(p3))].copy()
    for name, p in (('2', p2), ('3', p3)):
        keep['d' + name] = [p[x][0] for x in keep.pos]
        keep['a' + name] = [p[x][1] for x in keep.pos]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    keep[['pos', 'g2', 'g3', 'd2', 'a2', 'd3', 'a3']].to_csv(COUNTS, sep='\t', index=False)
    print(len(keep), 'sites with both libraries')


def load() -> tuple:
    d = pd.read_csv(COUNTS, sep='\t')
    m2, m3 = d.d2.median(), d.d3.median()
    d = d[(d.d2.between(0.5 * m2, 2 * m2)) & (d.d3.between(0.5 * m3, 2 * m3))].reset_index(drop=True)
    return d, m2, m3


def mix(rng, g, f: float, depth_target: int, m2: float, m3: float):
    """Add HG002 reads (share 1-f) and HG003 reads (share f) thinned to a mean total depth; f = 0 uses HG002 alone."""
    p2, p3 = min(1.0, (1 - f) * depth_target / m2), min(1.0, f * depth_target / m3)
    a = rng.binomial(g.a2.values, p2)
    d = a + rng.binomial(g.d2.values - g.a2.values, p2)                       # thin alt and ref reads jointly
    if f:
        a3 = rng.binomial(g.a3.values, p3)
        a, d = a + a3, d + a3 + rng.binomial(g.d3.values - g.a3.values, p3)
    return a, d


def draw_class(pool: pd.DataFrame, rng, n: int, f: float, depth_target: int, m2: float, m3: float):
    take = pool.sample(n, random_state=int(rng.integers(1 << 31)))
    return mix(rng, take, f, depth_target, m2, m3)


def run(budget: float) -> None:
    import importlib
    import screen_compare as sc
    model = importlib.import_module('partition_model_v3' if MODEL == 'v3' else 'partition_model_v3_2')
    if MODEL != 'v3':
        model.RHO = MODEL_RHO
    d, m2, m3 = load()
    pools = {k: g for k, g in d.groupby(['g2', 'g3'])}
    done = {json.loads(l)['key'] for l in RUNS.read_text().splitlines() if l.strip()} if RUNS.exists() else set()
    cells = [(f'pair{n}', f, (c, s)) for n, c, s, fs in PAIRS for f in fs] + [('control', f, (c,)) for f, c in CONTROLS]
    t0 = time.time()
    for rep in range(REPS):
        for kind, f, spec in cells:
            for dt in DEPTHS:
                key = f'{kind}|f{f}|d{dt}|r{rep}'
                if key in done:
                    continue
                if time.time() - t0 > budget:
                    print('budget reached')
                    return
                rng = np.random.default_rng([2929, rep, dt, int(f * 100), len(kind)])
                n = 2 * N_PER_CLASS if kind == 'control' else N_PER_CLASS
                parts = [draw_class(pools[g], rng, n, f, dt, m2, m3) for g in spec]
                keep = [p[1] >= MIN_DEPTH for p in parts]
                parts = [(p[0][k], p[1][k]) for p, k in zip(parts, keep)]
                alt, depth = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
                ones = np.ones(len(alt), int)
                res = model.fit_predict(alt, depth, ones, ones, 1.0, seed=rep)
                labels = np.asarray(res['labels'], int)
                cp = np.asarray(res['centers'], float)[labels]
                split = len(set(labels.tolist())) >= 2
                scores = sc.rule_scores(labels, cp, depth) if split else dict(gap=float('inf'), x_rho0=float('inf'), x_rho02=float('inf'))
                cls = [dict(vaf=float(p[0].sum() / p[1].sum()), rho=float(fit_rho(p[0].astype(float), p[1].astype(float))[1])) for p in parts]
                rec = dict(key=key, kind=kind, f=f, depth_target=dt, rep=rep, mean_depth=float(depth.mean()), k=int(res['k']), split=bool(split),
                           classes=cls, **{k: (None if np.isinf(v) else float(v)) for k, v in scores.items()})
                with RUNS.open('a') as fh:
                    fh.write(json.dumps(rec) + '\n')
                print(key, rec['k'], flush=True)
    print('ALL DONE')


def kept(g) -> str:
    s = g[g.split]
    return f'{int((s.x_rho02.fillna(np.inf) >= 1).sum())}/{len(s)}'


def report() -> None:
    runs = pd.DataFrame([json.loads(l) for l in RUNS.read_text().splitlines() if l.strip()])
    out = ['# Real-read check on computational mixtures of GIAB HG002 + HG003 (auto-generated by scripts/giab_mixture_check.py)', '',
           'Illumina 60x reads of two real cell lines, thinned and added at a chosen HG003 fraction f; chr20:1-30 Mb; copy-neutral, purity 1. '
           'x uses the observed class VAFs, the mean depth and the beta-binomial rho fitted to the drawn counts. Not a tumour and not a physical mixture.', '',
           '## Two-cluster mixtures: observed split rate against the frozen law', '',
           '| pair | f | target depth | mean depth | runs | clonal VAF | subclone VAF | rho (mean MLE) | x | split rate [95% CI] | law predicts | splits kept by x>=1 (rho 0.02) |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for n, _, _, fs in PAIRS:
        for f in fs:
            for dt in DEPTHS:
                g = runs[(runs.kind == f'pair{n}') & (runs.f == f) & (runs.depth_target == dt)]
                if not len(g):
                    continue
                q1, q2 = np.mean([c[0]['vaf'] for c in g.classes]), np.mean([c[1]['vaf'] for c in g.classes])
                rho = np.mean([(c[0]['rho'] + c[1]['rho']) / 2 for c in g.classes])
                md = g.mean_depth.mean()
                x = (q1 - q2) / np.sqrt(q1 * (1 - q1) * (1 / md + rho))
                pred = 1 / (1 + np.exp(-(B0 + B1 * np.log(x)))) if x > 0 else float('nan')
                lo, hi = wilson(int(g.split.sum()), len(g))
                out.append(f'| {n} | {f} | {dt} | {md:.0f} | {len(g)} | {q1:.3f} | {q2:.3f} | {rho:.4f} | {x:.2f} | {g.split.mean():.2f} [{lo:.2f}, {hi:.2f}] | {pred:.2f} | {kept(g)} |')
    out += ['', '## Single-class controls (every split is false)', '',
            '| f | target depth | mean depth | runs | VAF | rho (mean MLE) | false-split rate [95% CI] | splits kept by x>=1 (rho 0.02) |', '|---|---|---|---|---|---|---|---|']
    for f, _ in CONTROLS:
        for dt in DEPTHS:
            g = runs[(runs.kind == 'control') & (runs.f == f) & (runs.depth_target == dt)]
            if not len(g):
                continue
            lo, hi = wilson(int(g.split.sum()), len(g))
            out.append(f'| {f} | {dt} | {g.mean_depth.mean():.0f} | {len(g)} | {np.mean([c[0]["vaf"] for c in g.classes]):.3f} | '
                       f'{np.mean([c[0]["rho"] for c in g.classes]):.4f} | {g.split.mean():.2f} [{lo:.2f}, {hi:.2f}] | {kept(g)} |')
    out += summary(runs) + bootstrap(runs)
    REPORT.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print('\n'.join(out))


def summary(runs: pd.DataFrame) -> list:
    """Pooled screen outcome and error of the frozen law over the two-cluster cells."""
    rows = []
    for (kind, f, dt), g in runs[runs.kind.str.startswith('pair')].groupby(['kind', 'f', 'depth_target']):
        q1, q2 = np.mean([c[0]['vaf'] for c in g.classes]), np.mean([c[1]['vaf'] for c in g.classes])
        rho = np.mean([(c[0]['rho'] + c[1]['rho']) / 2 for c in g.classes])
        x = (q1 - q2) / np.sqrt(q1 * (1 - q1) * (1 / g.mean_depth.mean() + rho))
        rows.append(dict(x=x, obs=g.split.mean(), pred=1 / (1 + np.exp(-(B0 + B1 * np.log(x)))), splits=int(g.split.sum()),
                         kept=int(((g.x_rho02.fillna(np.inf) >= 1) & g.split).sum()), runs=len(g)))
    c = pd.DataFrame(rows)
    ctl = runs[runs.kind == 'control']
    cs, ck = int(ctl.split.sum()), int(((ctl.x_rho02.fillna(np.inf) >= 1) & ctl.split).sum())
    lines = ['', '## Pooled summary', '',
             f'- Controls: {cs} false splits in {len(ctl)} single-class runs ({cs / len(ctl):.0%}); the x >= 1 screen (rho assumed 0.02) keeps {ck} of them.']
    for name, m in (('x < 1', c.x < 1), ('x >= 1', c.x >= 1)):
        s = c[m]
        lines.append(f'- Two-cluster cells with {name} ({len(s)} cells, {int(s.runs.sum())} runs): {int(s.splits.sum())} splits, the screen keeps {int(s.kept.sum())}.')
    for name, m in (('all 16 cells', c.x > 0), ('x >= 0.6', c.x >= 0.6), ('x < 0.6', c.x < 0.6)):
        s = c[m]
        lines.append(f'- Frozen law versus observed split rate, {name}: RMSE {np.sqrt(((s.obs - s.pred) ** 2).mean()):.3f}, bias (observed - predicted) {(s.obs - s.pred).mean():+.3f}.')
    return lines


def bootstrap(runs: pd.DataFrame, n_boot: int = 2000) -> list:
    """95% intervals for the pooled figures by resampling the repetitions (seeds, which every cell shares) with replacement.
    x and the law's prediction per cell are held at their full-data values; only the split counts are resampled."""
    reps = np.sort(runs.rep.unique())
    runs = runs.assign(kept=runs.split & (runs.x_rho02.fillna(np.inf) >= 1))
    ctl, pairs = [], []
    for (kind, f, dt), g in runs.groupby(['kind', 'f', 'depth_target']):
        by = g.set_index('rep').reindex(reps)
        assert not by.split.isna().any(), 'incomplete run'
        cell = dict(split=by.split.astype(float).values, kept=by.kept.astype(float).values)
        if kind == 'control':
            ctl.append(cell)
            continue
        q1, q2 = np.mean([c[0]['vaf'] for c in g.classes]), np.mean([c[1]['vaf'] for c in g.classes])
        rho = np.mean([(c[0]['rho'] + c[1]['rho']) / 2 for c in g.classes])
        x = (q1 - q2) / np.sqrt(q1 * (1 - q1) * (1 / g.mean_depth.mean() + rho))
        pairs.append(dict(cell, x=x, pred=1 / (1 + np.exp(-(B0 + B1 * np.log(x))))))

    def stats(w: np.ndarray) -> dict:
        n = w.sum()
        out = {'ctl_rate': np.mean([c['split'] @ w / n for c in ctl]),
               'ctl_kept': sum(c['kept'] @ w for c in ctl)}
        for name, sel in (('lo', [p for p in pairs if p['x'] < 1]), ('hi', [p for p in pairs if p['x'] >= 1])):
            out[f'kept_{name}'] = sum(p['kept'] @ w for p in sel) / max(sum(p['split'] @ w for p in sel), 1e-9)
        for name, sel in (('all', pairs), ('x>=0.6', [p for p in pairs if p['x'] >= 0.6]), ('x<0.6', [p for p in pairs if p['x'] < 0.6])):
            d = np.array([p['split'] @ w / n - p['pred'] for p in sel])
            out[f'rmse {name}'], out[f'bias {name}'] = np.sqrt((d ** 2).mean()), d.mean()
        return out

    rng = np.random.default_rng(7)
    full = stats(np.ones(len(reps)))
    draws = pd.DataFrame([stats(np.bincount(rng.integers(0, len(reps), len(reps)), minlength=len(reps)).astype(float)) for _ in range(n_boot)])
    lo, hi = draws.quantile(0.025), draws.quantile(0.975)
    lines = ['', f'## Seed-resampled 95% intervals ({n_boot} resamples of the {len(reps)} repetitions; x and the law held fixed)', '']
    for k in full:
        fmt = '{:.0f}' if k == 'ctl_kept' else '{:+.3f}' if k.startswith('bias') else '{:.3f}'
        lines.append(f'- {k}: ' + fmt.format(full[k]) + ' [' + fmt.format(lo[k]) + ', ' + fmt.format(hi[k]) + ']')
    return lines


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'run':
        run(float(sys.argv[2]))
    else:
        {'sites': make_sites, 'counts': parse_counts, 'report': report}[cmd]()
