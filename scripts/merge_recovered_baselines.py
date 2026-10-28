"""Merge recovered baseline runs into the preregistered ClonoGate comparison (originals are never overwritten).

ClonoGate is the method `partition_v3` (reports/v3_dev_P, reports/v3_holdout).  `adaptive_dispersion` is an earlier candidate that
the preregistered analysis treats as one more baseline.  The comparison logic is analyze_v3_holdout.py's (paired over designs where
both methods have a score, 10,000-resample bootstrap over designs, preregistered seed, the same draw order), applied to a baseline
table.  Step 1 re-runs it on the ORIGINAL table and asserts that it reproduces reports/v3_holdout_ANALYSIS.json (the numbers in the
report).  Step 2 applies it to the table with recovered runs filling designs where the original had no score.  PhylogicNDT with the
numerical root-bracket fallback is a separate labelled variant, never merged into the unmodified tool.
Writes reports/baseline_recovery_v1/recovered_scores.csv and reports/BASELINE_RECOVERY_RESULTS.md.  Usage: py -3 scripts/merge_recovered_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from analyze_v3_holdout import BASELINES, PRE, mine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports' / 'baseline_recovery_v1'
TASKS = ('2A', '1B', '1C')  # the original analysis' order; it fixes the bootstrap draw order
BRACKET, COMBINED = 'phylogic_bracket_fixed', 'phylogic_combined'
OI = 'object_integration_6184761'
OI_FIX, OI_COMBINED = 'oi_guarded', 'oi_combined'
RECOVERED = {  # baseline method -> (official-scores file, method name inside it)
    OI_FIX: (BASE / 'oi_guarded' / 'official_scores.json', OI),
    'fastclone_core': (BASE / 'full_frozen' / 'official_scores.json', 'fastclone_core'),
    'ccube_public': (BASE / 'scoring' / 'ccube' / 'official_scores.json', 'ccube_public'),
    'phylogic_public': (BASE / 'scoring' / 'phylogic' / 'official_scores.json', 'phylogic_public'),
    BRACKET: (BASE / 'scoring' / 'phylogic_bracketed' / 'official_scores.json', 'phylogic_public'),
}
NAMES = {'fastclone_core': 'FastClone', 'ccube_public': 'CCube', 'phylogic_public': 'PhylogicNDT',
         'object_integration_6184761': 'Object Integration', 'adaptive_dispersion': 'earlier candidate (adaptive dispersion)',
         BRACKET: 'PhylogicNDT, bracket-fixed variant',
         COMBINED: 'PhylogicNDT, unmodified where it ran + bracket-fixed variant where it crashed',
         OI_FIX: 'Object Integration, guarded variant',
         OI_COMBINED: 'Object Integration, unmodified where it ran + guarded variant where it crashed'}


def recovered_scores(path: Path, method: str) -> dict[tuple[str, str], float]:
    """Finite scored values of one recovery run, keyed (design, task)."""
    out = {}
    for r in json.loads(path.read_text()):
        score = r['score'][-1] if isinstance(r['score'], list) else r['score']
        if r['method'] == method and r['challenge'] in TASKS and r['status'] == 'scored' and np.isfinite(score):
            out[(r['sample'], r['challenge'])] = float(score)
    return out


def merged_table(original: pd.DataFrame) -> pd.DataFrame:
    """Original rows plus recovered fills, the bracket-fixed variant, and the combined PhylogicNDT column."""
    d = original.copy()
    for method in ('fastclone_core', 'ccube_public', 'phylogic_public'):
        rec = recovered_scores(*RECOVERED[method])
        for i in d.index[(d.method == method) & d.score.isna()]:
            value = rec.get((d.at[i, 'sample'], d.at[i, 'task']))
            if value is not None:
                d.loc[i, ['score', 'status']] = [value, 'scored']
    parts = [d]
    for variant, base, combined in ((BRACKET, 'phylogic_public', COMBINED), (OI_FIX, OI, OI_COMBINED)):
        lacking = d[(d.method == base) & d.score.isna()]
        missing = set(zip(lacking['sample'], lacking['task']))  # a variant only counts where the original run has no score
        fix = {k: v for k, v in recovered_scores(*RECOVERED[variant]).items() if k in missing}
        parts.append(pd.DataFrame([dict(sample=s, method=variant, task=t, score=v, status='scored') for (s, t), v in fix.items()]))
        comb = d[d.method == base][['sample', 'task', 'score', 'status']].copy()
        need = comb.score.isna()
        comb.loc[need, 'score'] = [fix.get((s, t), np.nan) for s, t in zip(comb['sample'][need], comb['task'][need])]
        comb.loc[comb.score.notna(), 'status'] = 'scored'
        parts.append(comb.assign(method=combined))
    return pd.concat(parts, ignore_index=True)


def compare(scores: dict, designs: list[str], label: str, rng: np.random.Generator, table: pd.DataFrame, methods: list[str]) -> dict:
    """analyze_v3_holdout.compare on a given baseline table and method list (logic and draw order unchanged)."""
    b = table[table.score.notna()]
    res = {}
    for task in TASKS:
        have = [s for s in designs if (s, task) in scores]
        res[task] = dict(label=label, designs=len(designs), v3_scored=len(have), vs={})
        for m in methods:
            bb = b[(b.method == m) & (b.task == task)].set_index('sample').score
            pairs = [(scores[(s, task)], float(bb[s])) for s in have if s in bb.index]
            if len(pairs) < 3:
                continue
            d = np.array([x - y for x, y in pairs])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(PRE['bootstrap_reps'])]
            res[task]['vs'][m] = dict(n=len(d), v3=float(np.mean([x for x, _ in pairs])), baseline=float(np.mean([y for _, y in pairs])),
                                      diff=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                      wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()))
    return res


def analyse(table: pd.DataFrame, hold: dict, dev: dict, methods: list[str]) -> dict:
    """Holdout (26 preregistered designs) and all-51 comparisons, drawing from one generator as the original did."""
    rng = np.random.default_rng(PRE['bootstrap_seed'])
    designs = PRE['holdout_designs'] + sorted({s for s, _ in dev})
    return dict(holdout=compare(hold, PRE['holdout_designs'], 'holdout', rng, table, methods),
                all=compare({**dev, **hold}, designs, 'all', rng, table, methods))


def check_reproduces(original_result: dict) -> None:
    """The draft's numbers (reports/v3_holdout_ANALYSIS.json) must come back exactly from the original table."""
    saved = json.loads((ROOT / 'reports' / 'v3_holdout_ANALYSIS.json').read_text())
    for scope, key in (('holdout', 'holdout'), ('all', 'all_designs_dev_flagged')):
        for task in TASKS:
            for m, x in saved[key][task]['vs'].items():
                y = original_result[scope][task]['vs'][m]
                assert y['n'] == x['n'] and abs(y['diff'] - x['diff']) < 1e-9 and np.allclose(y['ci'], x['ci'], atol=1e-9), (scope, task, m)


def paired_lines(result: dict, methods: list[str]) -> list[str]:
    lines = ['| Designs | Comparator | Task | Pairs | ClonoGate | Comparator | Difference [95% interval] | Wins / losses |',
             '|---|---|---|---:|---:|---:|---|---:|']
    for scope, name in (('holdout', 'Holdout (26)'), ('all', 'All 51')):
        for m in methods:
            for task in TASKS:
                x = result[scope][task]['vs'].get(m)
                if x:
                    lines.append(f"| {name} | {NAMES[m]} | {task} | {x['n']} | {x['v3']:.4f} | {x['baseline']:.4f} | {x['diff']:+.4f} "
                                 f"[{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] | {x['wins']} / {x['losses']} |")
    return lines


def coverage_lines(before: pd.DataFrame, after: pd.DataFrame) -> list[str]:
    lines = ['| Method | 1B before | 1B after | 1C before | 1C after | 2A before | 2A after |', '|---|---:|---:|---:|---:|---:|---:|']
    for m in BASELINES[:4] + [BRACKET, COMBINED, OI_FIX, OI_COMBINED]:
        cells = []
        for t in ('1B', '1C', '2A'):
            count = lambda d: str(int(d[(d.method == m) & (d.task == t)].score.notna().sum()))  # noqa: E731
            cells += [count(before) if m in set(before.method) else '-', count(after)]
        lines.append(f'| {NAMES[m]} | ' + ' | '.join(cells) + ' |')
    return lines


def unrecovered_lines(after: pd.DataFrame) -> list[str]:
    lines = []
    for m in BASELINES[:4]:
        g = after[(after.method == m) & (after.task == '2A') & after.score.isna()]
        reasons = g.groupby('status')['sample'].apply(lambda s: ', '.join(sorted(s)))
        lines.append(f'- **{NAMES[m]}** ({len(g)} designs without a 2A score): ' + '; '.join(f'{k}: {v}' for k, v in reasons.items()))
    return lines


def main() -> None:
    original = pd.read_csv(ROOT / 'reports' / 'full_benchmark_scores.csv')
    hold, dev = mine(ROOT / 'reports' / 'v3_holdout' / 'official_scores.json'), mine(ROOT / 'reports' / 'v3_dev_P' / 'official_scores.json')
    before = analyse(original, hold, dev, BASELINES)
    check_reproduces(before)
    after_table = merged_table(original)
    after_table.to_csv(BASE / 'recovered_scores.csv', index=False)
    after = analyse(after_table, hold, dev, BASELINES)
    variants = analyse(after_table, hold, dev, [BRACKET, COMBINED, OI_FIX, OI_COMBINED])
    lines = ['# Missing-baseline recovery: coverage and ClonoGate paired comparison (auto-generated by scripts/merge_recovered_baselines.py)', '',
             'ClonoGate is `partition_v3`. The "before" table is the preregistered analysis re-run on the original baseline scores; the script '
             'asserts that it reproduces `reports/v3_holdout_ANALYSIS.json` (the numbers in the report) before anything else is computed. '
             'The "after" table is the same analysis with recovered runs filling designs where the original baseline had no score. '
             'This is a post-hoc update of baseline coverage; the preregistered holdout result is the "before" table. Scores are the official '
             'SMC-Het scorer, unmodified. Recovered CCube and PhylogicNDT runs use the original command lines with longer wall-clock caps '
             '(CCube 6 h, PhylogicNDT 10 h). Recovered FastClone assigns the 1-4 mutations per design whose posterior row is undefined to the '
             'largest cluster. The bracket-fixed PhylogicNDT is a separate variant that widens only a missing Stirling root bracket.', '',
             '## Coverage (designs scored, of 51)', ''] + coverage_lines(original, after_table)
    lines += ['', '## Designs still without a score (2A), by original failure status', ''] + unrecovered_lines(after_table)
    lines += ['', '## Before recovery (preregistered analysis, original baseline scores)', ''] + paired_lines(before, BASELINES)
    lines += ['', '## After recovery', ''] + paired_lines(after, BASELINES)
    lines += ['', '## PhylogicNDT and Object Integration variants (after recovery)', ''] + paired_lines(variants, [BRACKET, COMBINED, OI_FIX, OI_COMBINED]) + ['']
    (ROOT / 'reports' / 'BASELINE_RECOVERY_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
