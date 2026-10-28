"""Screen comparison (experiment 1): is dividing the prevalence gap by estimated noise better than simpler rules?

Three rules score every reported subclone call (K >= 2) from the method's own clusters (clusters holding at least 2% of SNVs,
cluster medians; calls with fewer than two such clusters score inf and are always kept):
  gap       raw prevalence gap c1 - c2
  x_rho0    standardised gap with binomial noise only (assumed rho 0)
  x_rho02   standardised gap with assumed rho .02
A call is dropped when its score is below a threshold.  For each rule and tool the threshold is swept and we report the share of
false splits removed (noisy single-clone tumours) at matched shares of true detections lost (two-clone tumours), for scope 'all'
(the 135-tumour pool of reporting_rule.py) and scope 'grid' (equal clones only, where depth and purity vary).
Writes reports/v3_falsesplit/SCREEN_COMPARE.md.  Usage: screen_compare.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from reporting_rule import MIN_CLUSTER_FRAC, category  # noqa: E402
from score_accuracy import DATASETS, GRID_SEEDS, NAMES, PROJECT, REPORTS, ccube_result, read_jsonl, truth  # noqa: E402

OUT = REPORTS / 'v3_falsesplit' / 'SCREEN_COMPARE.md'
RULES = ('gap', 'x_rho0', 'x_rho02')
RHO_BY_RULE = {'x_rho0': 0.0, 'x_rho02': 0.02}
LOSS_TARGETS = (0.10, 0.18, 0.30)
SCOPES = {'all': None, 'grid': 'grid'}
SKIP_METHODS = ('myclone', 'fastclone')  # MyClone splits come from its multiplicity convention; FastClone has too few splits


def rule_scores(labels: np.ndarray, cp: np.ndarray, depth: np.ndarray) -> dict:
    """Score of each rule for one call; inf (always kept) when fewer than two sizeable clusters."""
    centres = sorted((float(np.median(cp[labels == k])) for k in np.unique(labels)
                      if (labels == k).mean() >= MIN_CLUSTER_FRAC), reverse=True)
    if len(centres) < 2:
        return {rule: float('inf') for rule in RULES}
    gap = centres[0] - centres[1]
    q, dq = centres[0] / 2, gap / 2
    scores = {'gap': gap}
    for rule, rho in RHO_BY_RULE.items():
        scores[rule] = float(dq / np.sqrt(q * (1 - q) * (1 / depth.mean() + rho)))
    return scores


def collect(scope_dataset: str | None) -> dict:
    """{(method, 'false'|'detect'): [score dict per reported split]} over noisy single-clone and two-clone tumours."""
    calls: dict = defaultdict(list)
    for dataset in DATASETS:
        if scope_dataset and dataset != scope_dataset:
            continue
        data = PROJECT / 'data' / DATASETS[dataset]
        manifest = [m for m in read_jsonl(data / 'manifest.jsonl') if m['seed'] < GRID_SEEDS]
        sources = {meth: {r['id']: r for r in read_jsonl(REPORTS / 'extra' / f'{dataset}__{meth}.jsonl')}
                   for meth in NAMES if meth != 'ccube'}
        for m in manifest:
            cat = category(dataset, m)
            if cat == 'single, rho 0':
                continue
            kind = 'detect' if cat == 'two-clone' else 'false'
            _, _, depth, _ = truth(dataset, m)
            found = {'ccube': ccube_result(dataset, m['id'])}
            for meth, rows in sources.items():
                r = rows.get(m['id'])
                if r is not None and r['status'] == 'ok':
                    found[meth] = (np.array(r['labels'], int), np.array(r['cp'], float), r['k'])
            for meth, res in found.items():
                if res is not None and res[2] >= 2 and meth not in SKIP_METHODS:
                    calls[(meth, kind)].append(rule_scores(res[0], res[1], depth))
    return calls


def removed_at_loss(false_scores: np.ndarray, detect_scores: np.ndarray, target: float) -> float:
    """Largest share of false splits removed by any threshold that loses at most `target` of the true detections."""
    false_sorted, detect_sorted = np.sort(false_scores), np.sort(detect_scores)
    thresholds = np.unique(np.concatenate([false_sorted, detect_sorted]))
    removed = np.searchsorted(false_sorted, thresholds, side='left') / len(false_sorted)
    lost = np.searchsorted(detect_sorted, thresholds, side='left') / len(detect_sorted)
    return float(removed[lost <= target].max(initial=0.0))


def x_one_point(false_scores: np.ndarray, detect_scores: np.ndarray) -> tuple[float, float]:
    """(false splits removed, detections lost) for the fixed rule score >= 1."""
    return float((false_scores < 1).mean()), float((detect_scores < 1).mean())


def render_scope(name: str, calls: dict) -> list[str]:
    lines = [f'## Scope: {name}', '',
             'Share of false splits removed at matched detection loss: raw gap / x (rho 0) / x (rho .02).  '
             'Last columns: the fixed x >= 1 rule (removed / lost) at assumed rho 0 and .02.', '',
             '| tool | false / true splits | loss <= 10% | loss <= 18% | loss <= 30% | x>=1 at rho 0 | x>=1 at rho .02 |',
             '|---|---|---|---|---|---|---|']
    for meth, label in NAMES.items():
        false, detect = calls.get((meth, 'false')), calls.get((meth, 'detect'))
        if not false or not detect:
            continue
        fs = {rule: np.array([c[rule] for c in false]) for rule in RULES}
        ds = {rule: np.array([c[rule] for c in detect]) for rule in RULES}
        cells = [' / '.join(f'{100 * removed_at_loss(fs[r], ds[r], t):.0f}%' for r in RULES) for t in LOSS_TARGETS]
        refs = [' / '.join(f'{100 * v:.0f}%' for v in x_one_point(fs[r], ds[r])) for r in ('x_rho0', 'x_rho02')]
        lines.append(f'| {label} | {len(false)} / {len(detect)} | ' + ' | '.join(cells + refs) + ' |')
    return lines + ['']


def main() -> None:
    body = ['# Screen comparison: raw gap vs standardised gap (auto-generated by scripts/screen_compare.py)', '',
            'Noisy single-clone tumours supply the false splits; two-clone tumours supply the true detections (same pools as '
            'REPORTING_RULE.md).  Thresholds are swept per tool, so the comparison is at matched detection loss, not at one tuned '
            'cutoff.  The sweep picks the best threshold per tool on the same data it is scored on, so all three rules are given '
            'an optimistic in-sample advantage.', '']
    for name, dataset in SCOPES.items():
        body += render_scope(name, collect(dataset))
    OUT.write_text('\n'.join(body), encoding='utf-8')
    print('\n'.join(body))


if __name__ == '__main__':
    main()
