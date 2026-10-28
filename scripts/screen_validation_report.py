"""Report for the independent validation of the x >= 1 screen and the wider oracle comparison (new simulation family).

Rules, fixed in data/sim_screen/plan.json before any fit: keep a reported split only if x >= 1 (x from the tool's own clusters, assumed
rho 0 or 0.02), or only if the plain prevalence gap c1 - c2 is at least a cutoff tuned ONLY on the earlier pooled simulations (the
cutoff removing the most false splits at <= 18% loss of true splits).  A second check tunes the cutoff on the fitted-simulator (bb)
half of the new family and tests it on the held-out-generator (ho) half.  Metrics: false splits removed (single-clone tumours), true
splits lost (two-clone tumours), mean ARI over two-clone tumours with each rule applied (a dropped split is one cluster, ARI 0), fraction
of the oracle ARI.  Intervals: bootstrap of tumours within settings, 2,000 replicates.  Writes reports/v3_falsesplit/SCREEN_VALIDATION.md.
Usage: py -3 scripts/screen_validation_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import betabinom, binom

sys.path.append(str(Path(__file__).resolve().parent))
import screen_compare as sc  # noqa: E402
import screen_validation as sv  # noqa: E402
from score_accuracy import ari  # noqa: E402

OUT = sv.RESULTS / 'SCREEN_VALIDATION.md'
CCUBE_DIR = sv.ROOT / 'reports' / 'ccube_simscreen'
N_BOOT, BOOT_SEED, MAX_LOSS = 2000, 20261026, 0.18
TOOLS = {'clonogate': ('ClonoGate (frozen v3)', 'v3'), 'pyclone': ('PyClone-VI binomial', 'pyclone-binomial'), 'ccube': ('CCube', 'ccube')}
RULES = ('none', 'x_rho0', 'x_rho02', 'gap_tuned')
RULE_LABELS = {'none': 'no screen', 'x_rho0': 'x >= 1, rho 0', 'x_rho02': 'x >= 1, rho 0.02', 'gap_tuned': 'plain gap, tuned cutoff'}


def read_results() -> dict:
    """{tool: {id: (labels, cp)}} for finished fits."""
    out: dict = {t: {} for t in TOOLS}
    for tool, pattern in (('clonogate', 'screen_validation_v3.jsonl'), ('pyclone', 'screen_validation_pyclone-binomial_*.jsonl')):
        for path in sv.RESULTS.glob(pattern):
            for line in path.read_text().splitlines():
                r = json.loads(line) if line.strip() else None
                if r and r['status'] == 'ok':
                    out[tool][r['id']] = (np.array(r['labels'], int), np.array(r['cp'], float))
    for folder in CCUBE_DIR.glob('sv*/ccube_public'):
        if (folder / '2A.txt').exists() and (folder / '1C.txt').exists():
            labels = np.loadtxt(folder / '2A.txt', dtype=int, ndmin=1)
            table = np.loadtxt(folder / '1C.txt', ndmin=2)
            out['ccube'][folder.parent.name] = (labels, pd.Series(table[:, 2], index=table[:, 0].astype(int)).loc[labels].to_numpy())
    return out


def oracle_ari(t: dict, alt, depth, major, minor, sub) -> float:
    """Posterior-assignment oracle on copy-neutral SNVs: true prevalences, true mixing share, true rho (beta-binomial)."""
    keep = (major == 1) & (minor == 1)
    a, n = alt[keep], depth[keep]
    q_clonal, q_sub, rho = t['purity'] / 2, (1 - t['delta']) * t['purity'] / 2, t['rho']

    def lp(q):
        return betabinom.logpmf(a, n, q * (1 / rho - 1), (1 - q) * (1 / rho - 1)) if rho else binom.logpmf(a, n, q)
    pred = (np.log(t['frac_sub']) + lp(q_sub) > np.log(1 - t['frac_sub']) + lp(q_clonal)).astype(int)
    return ari(sub[keep].astype(int), pred)


def x_eff(t: dict) -> float:
    q = t['purity'] / 2
    x = t['delta'] * t['purity'] / 2 / np.sqrt(q * (1 - q) * (1 / t['depth'] + t['rho']))
    return float(x * (t['n_snv'] / 700) ** sv.FROZEN['a'] * (t['frac_sub'] / 0.5) ** sv.FROZEN['c'])


def tune_cut(false_scores: np.ndarray, detect_scores: np.ndarray) -> float:
    """Cutoff removing the most false splits while losing at most MAX_LOSS of the true splits (smallest such cutoff)."""
    false, detect = np.sort(false_scores), np.sort(detect_scores)
    thr = np.unique(np.concatenate([false, detect]))
    removed = np.searchsorted(false, thr, side='left') / len(false)
    lost = np.searchsorted(detect, thr, side='left') / len(detect)
    ok = lost <= MAX_LOSS
    return float(thr[ok][np.argmax(removed[ok])])


def old_pool_cuts() -> dict:
    """Plain-gap cutoff per tool tuned on the earlier pooled simulations only."""
    calls = sc.collect(None)
    return {tool: tune_cut(np.array([c['gap'] for c in calls[(m, 'false')]]), np.array([c['gap'] for c in calls[(m, 'detect')]]))
            for tool, (_, m) in TOOLS.items()}


def build(results: dict, cuts: dict) -> pd.DataFrame:
    """One row per (tool, tumour): split flag, gap and x scores, rule decisions, raw ARI, ARI under each rule, oracle ARI."""
    rows = []
    for t in sv.tumours(json.loads(sv.PLAN.read_text())):
        alt, depth, major, minor, sub = sv.simulate_truth(t)
        two = t['kind'] == 'two'
        oracle = oracle_ari(t, alt, depth, major, minor, sub) if two else np.nan
        for tool in TOOLS:
            res = results[tool].get(t['id'])
            if res is None:
                continue
            labels, cp = res
            split = len(set(labels.tolist())) >= 2
            scores = sc.rule_scores(labels, cp, depth) if split else dict(gap=np.inf, x_rho0=np.inf, x_rho02=np.inf)
            raw = ari(sub.astype(int), labels) if two else np.nan
            rows.append(dict(tool=tool, id=t['id'], setting=t['index'], kind=t['kind'], generator=t['generator'], depth=t['depth'],
                             purity=t['purity'], frac_sub=t['frac_sub'], rho=t['rho'], x_eff=x_eff(t) if two else np.nan, split=split,
                             gap=scores['gap'], x_rho0=scores['x_rho0'], x_rho02=scores['x_rho02'], ari_raw=raw, oracle=oracle))
    return apply_rules(pd.DataFrame(rows), cuts)


def apply_rules(d: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    """Add keep_<rule> and ari_<rule> columns; a dropped split is one cluster (ARI 0), and K = 1 fits are unchanged."""
    d = d.copy()
    d['keep_none'], d['keep_x_rho0'], d['keep_x_rho02'] = True, d.x_rho0 >= 1, d.x_rho02 >= 1
    d['keep_gap_tuned'] = d.gap >= d.tool.map(cuts)
    for r in RULES:
        dropped = d.split & ~d[f'keep_{r}']
        d[f'ari_{r}'] = np.where(d.kind == 'two', np.where(dropped, 0.0, d.ari_raw), np.nan)
    return d


def draws(setting: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap index matrix (N_BOOT x N): tumours resampled with replacement within each setting."""
    out, pos = np.empty((N_BOOT, len(setting)), int), 0
    for s in np.unique(setting):
        g = np.flatnonzero(setting == s)
        out[:, pos:pos + len(g)] = g[rng.integers(0, len(g), (N_BOOT, len(g)))]
        pos += len(g)
    return out


def ratio_ci(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> str:
    """'point [lo, hi]' of sum(num)/sum(den), intervals from the bootstrap index matrix."""
    point = num.sum() / den.sum() if den.sum() else np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        reps = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    if np.isnan(point) or np.isnan(reps).all():
        return '-'
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return f'{point:.2f} [{lo:.2f}, {hi:.2f}]'


def screen_table(d: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    """Per tool and rule: false splits before -> after and removed share, true splits before -> after and lost share, ARI, oracle fraction."""
    lines = ['| tool | rule | single-clone tumours with a split, before -> after | false splits removed | two-clone tumours with a split, before -> after '
             '| true splits lost | mean ARI, two-clone | fraction of oracle |', '|---|---|---|---|---|---|---|---|']
    for tool, (label, _) in TOOLS.items():
        t = d[d.tool == tool].reset_index(drop=True)
        if t.empty:
            continue
        idx = draws(t.setting.to_numpy(), rng)
        single, two, split = (t.kind == 'single').to_numpy(), (t.kind == 'two').to_numpy(), t.split.to_numpy()
        for r in RULES:
            keep = t[f'keep_{r}'].to_numpy()
            f_rep, f_after = (single & split).astype(float), (single & split & keep).astype(float)
            t_rep, t_after = (two & split).astype(float), (two & split & keep).astype(float)
            ari_r = np.nan_to_num(t[f'ari_{r}'].to_numpy()) * two
            oracle = np.nan_to_num(t.oracle.to_numpy()) * two
            lines.append(f'| {label} | {RULE_LABELS[r]} | {int(f_rep.sum())} -> {int(f_after.sum())} of {int(single.sum())} | '
                         f'{ratio_ci(f_rep - f_after, f_rep, idx)} | {int(t_rep.sum())} -> {int(t_after.sum())} of {int(two.sum())} | '
                         f'{ratio_ci(t_rep - t_after, t_rep, idx)} | {ratio_ci(ari_r, two.astype(float), idx)} | {ratio_ci(ari_r, oracle, idx)} |')
    return lines + ['']


def matched_loss_table(d: pd.DataFrame) -> list[str]:
    """Share of false splits removed at matched detection loss, thresholds swept on the new family itself (in-sample for every rule)."""
    lines = ['| tool | false / true splits | loss <= 10% | loss <= 18% | loss <= 30% |', '|---|---|---|---|---|']
    for tool, (label, _) in TOOLS.items():
        t = d[d.tool == tool]
        false, detect = t[(t.kind == 'single') & t.split], t[(t.kind == 'two') & t.split]
        if false.empty or detect.empty:
            continue
        cells = [' / '.join(f'{100 * sc.removed_at_loss(false[r].to_numpy(), detect[r].to_numpy(), target):.0f}%'
                            for r in ('gap', 'x_rho0', 'x_rho02')) for target in (0.10, 0.18, 0.30)]
        lines.append(f'| {label} | {len(false)} / {len(detect)} | ' + ' | '.join(cells) + ' |')
    return lines + ['']


def factor_table(d: pd.DataFrame, factor: str, bins, rng: np.random.Generator) -> list[str]:
    """Fraction of the oracle ARI per factor level: each tool without a screen, and ClonoGate with the x >= 1 (rho 0.02) screen."""
    cols = [('clonogate', 'none', 'ClonoGate'), ('clonogate', 'x_rho02', 'ClonoGate + screen (rho 0.02)'),
            ('pyclone', 'none', 'PyClone-VI binomial'), ('ccube', 'none', 'CCube')]
    lines = [f'| {factor} | two-clone tumours | oracle mean ARI | ' + ' | '.join(c[2] for c in cols) + ' |', '|---|---|---|' + '---|' * len(cols)]
    sub = {tool: d[(d.tool == tool) & (d.kind == 'two')].reset_index(drop=True) for tool in TOOLS}
    idxs = {tool: draws(t.setting.to_numpy(), rng) for tool, t in sub.items() if not t.empty}
    for label, test in bins:
        base = sub['clonogate']
        sel = test(base)
        cells = []
        for tool, rule, _ in cols:
            if tool not in idxs:
                cells.append('-')
                continue
            t = sub[tool]
            m = test(t).to_numpy().astype(float)
            cells.append(ratio_ci(np.nan_to_num(t[f'ari_{rule}'].to_numpy()) * m, np.nan_to_num(t.oracle.to_numpy()) * m, idxs[tool]))
        lines.append(f'| {label} | {int(sel.sum())} | {base[sel].oracle.mean():.3f} | ' + ' | '.join(cells) + ' |')
    return lines + ['']


def main() -> None:
    rng = np.random.default_rng(BOOT_SEED)
    cuts = old_pool_cuts()
    results = read_results()
    d = build(results, cuts)
    lines = ['# Independent validation of the x >= 1 screen, and wider oracle comparison (auto-generated by scripts/screen_validation_report.py)', '',
             'New simulation family (data/sim_screen/plan.json, written before any fit): 80 settings x 12 tumours, each tumour with its own new '
             'seed; 40 single-clone (false splits) and 40 two-clone (true splits) settings, half from the fitted simulator (bb) and half from '
             'the held-out generator (ho). Fits finished: ' + ', '.join(f'{TOOLS[t][0]} {len(r)}' for t, r in results.items()) + '.', '',
             f'Plain-gap cutoffs tuned only on the earlier pooled simulations (at most {MAX_LOSS:.0%} of true splits lost): ' +
             ', '.join(f'{TOOLS[t][0]} {g:.3f}' for t, g in cuts.items()) + '.', '']
    for name, test in (('all tumours', lambda x: x.id.notna()), ('fitted simulator (bb)', lambda x: x.generator == 'bb'),
                       ('held-out generator (ho)', lambda x: x.generator == 'ho')):
        lines += [f'## 1. Screen rules on the new family: {name}', ''] + screen_table(d[test(d)], rng)
    bb = d[d.generator == 'bb']
    cuts_bb = {tool: tune_cut(bb[(bb.tool == tool) & (bb.kind == 'single') & bb.split].gap.to_numpy(),
                              bb[(bb.tool == tool) & (bb.kind == 'two') & bb.split].gap.to_numpy())
               for tool in TOOLS if ((bb.tool == tool) & (bb.kind == 'single') & bb.split).any() and ((bb.tool == tool) & (bb.kind == 'two') & bb.split).any()}
    ho = apply_rules(d[d.generator == 'ho'], {**cuts, **cuts_bb})
    lines += ['## 2. Second check: plain-gap cutoff tuned on the fitted-simulator (bb) half of the new family, tested on the held-out-generator (ho) half', '',
              'Cutoffs tuned on the bb half: ' + ', '.join(f'{TOOLS[t][0]} {g:.3f}' for t, g in cuts_bb.items()) + '.', ''] + screen_table(ho, rng)
    lines += ['## 2b. Trade-off at matched detection loss on the new family (raw gap / x at rho 0 / x at rho 0.02)', '',
              'Largest share of false splits removed by any threshold that loses at most the stated share of true splits; each rule\'s threshold is '
              'swept on these same tumours, so every rule gets an in-sample advantage (the same convention as SCREEN_COMPARE.md).', ''] + matched_loss_table(d)
    lines += ['## 3. Oracle comparison by factor (two-clone tumours)', '',
              'Fraction of the oracle ARI = mean ARI divided by the oracle\'s mean ARI over the same tumours, 95% bootstrap intervals; the oracle '
              'assigns copy-neutral SNVs at the true prevalences, mixing share and rho (an approximation for the held-out generator).', '']
    for factor, vals in (('depth', sv.DEPTHS), ('purity', sv.PURITIES), ('frac_sub', sv.FRACS), ('rho', sv.RHOS)):
        lines += factor_table(d, factor, [(f'{factor} {v}', lambda x, v=v, f=factor: x[f] == v) for v in vals], rng)
    lines += factor_table(d, 'x_eff', [('x_eff < 0.7', lambda x: x.x_eff < 0.7), ('0.7 - 1.0', lambda x: (x.x_eff >= 0.7) & (x.x_eff < 1.0)),
                                       ('1.0 - 1.4', lambda x: (x.x_eff >= 1.0) & (x.x_eff < 1.4)), ('>= 1.4', lambda x: x.x_eff >= 1.4)], rng)
    lines += factor_table(d, 'generator', [('bb', lambda x: x.generator == 'bb'), ('ho', lambda x: x.generator == 'ho')], rng)
    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
