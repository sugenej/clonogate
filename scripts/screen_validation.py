"""Independent validation of the x >= 1 screen, and a wider oracle comparison, on a NEW simulation family (item 1 / item 5).

`plan` draws 80 settings (40 single-clone, 40 two-clone; half from the fitted simulator 'bb', half from the held-out generator 'ho')
x 12 tumours, every tumour with its own new seed (7000 + index, disjoint from all earlier seeds), and writes the settings AND the
pre-specified analysis (rules, tuning protocol, metrics) to data/sim_screen/plan.json before anything is fitted.  Two-clone gaps are
set so the folded standardised gap x_eff of the frozen law is spread over 0.4-2.0, which straddles the screen's threshold.
`export` writes PyClone-VI and CCube input tables for each tumour (same layouts as data/sim_grid; CCube only for the first 3
tumours of each setting, because it takes ~110 s per tumour).  `run-v3` fits frozen ClonoGate (partition_model_v3) in-process;
`run-pyclone` fits PyClone-VI (binomial).  All runs are resumable and keep labels and per-SNV prevalences for scoring.
Usage: screen_validation.py plan | export | run-v3 | run-pyclone [SHARD NSHARDS]
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'model'))

DATA = ROOT / 'data' / 'sim_screen'
PLAN = DATA / 'plan.json'
RESULTS = ROOT / 'reports' / 'v3_falsesplit'
N_SINGLE = N_TWO = 40
N_SEEDS, CCUBE_SEEDS = 12, 3
PLAN_SEED, SEED_BASE = 20261026, 7000
GENERATORS, GAIN_FRAC = ('bb', 'ho'), {'bb': 0.2, 'ho': 0.35}
DEPTHS, PURITIES, RHOS = (40, 60, 100, 150), (0.4, 0.55, 0.7, 0.8), (0.0, 0.005, 0.01, 0.02)
N_SNVS, FRACS = (500, 1000), (0.2, 0.35, 0.5)
FROZEN = dict(a=0.24, c=0.38)  # exponents of the frozen 192-cell law (x_eff = x (N/700)^a (fs/0.5)^c)
NB_SIZE = 8.0

ANALYSIS = {
    'rules': ('Keep a reported subclone (K >= 2) only if x >= 1, x from the tool own clusters (>= 2% of SNVs, cluster medians, '
              'highest vs next-highest prevalence) with assumed rho 0 or 0.02 (never the true rho).  A dropped call becomes K = 1.'),
    'plain_gap': ('Plain prevalence-gap cutoff c1 - c2, tuned ONLY on the earlier pooled simulations (screen_compare pool, '
                  'frozen ClonoGate): the threshold that removes the most false splits while losing at most 18% of true splits '
                  '(the loss the x >= 1 screen cost at assumed rho 0.02 there).  Frozen before the new family is scored.'),
    'metrics': ('Per tool: false splits among single-clone tumours (reported, removed by each rule); true splits among two-clone '
                'tumours (reported, lost by each rule); mean ARI against the true clonal/subclonal split over two-clone tumours with '
                'each rule applied (a dropped call is one cluster, ARI 0); fraction of the oracle ARI.'),
    'intervals': 'Bootstrap over tumours within settings (each tumour has its own seed), 2,000 replicates, seed 20261026.',
    'oracle': ('Oracle: knows true cluster prevalences, mixing share and true rho; maximum-posterior assignment of copy-neutral SNVs '
               'under a beta-binomial at the true rho (for generator ho, whose noise is logit-normal, this is an approximation).'),
    'subsets': 'ClonoGate and PyClone-VI on all tumours; CCube on the first 3 tumours of each setting.',
}


def make_plan() -> dict:
    """80 settings; two-clone gaps solved from a target x_eff drawn uniformly in 0.4-2.0."""
    rng, settings = np.random.default_rng(PLAN_SEED), []
    while len(settings) < N_SINGLE + N_TWO:
        i = len(settings)
        s = dict(index=i, kind='single' if i % 2 == 0 else 'two', generator=GENERATORS[(i // 2) % 2], rho=float(rng.choice(RHOS)),
                 depth=int(rng.choice(DEPTHS)), purity=float(rng.choice(PURITIES)), n_snv=int(rng.choice(N_SNVS)),
                 frac_sub=float(rng.choice(FRACS)), delta=0.0)
        if s['kind'] == 'two':
            fold = (s['n_snv'] / 700) ** FROZEN['a'] * (s['frac_sub'] / 0.5) ** FROZEN['c']
            q = s['purity'] / 2
            s['delta'] = round(2 * (rng.uniform(0.4, 2.0) / fold) * np.sqrt(q * (1 - q) * (1 / s['depth'] + s['rho'])) / s['purity'], 3)
            if not 0.05 <= s['delta'] <= 0.9:
                continue
        settings.append(s)
    return dict(plan_seed=PLAN_SEED, seed_base=SEED_BASE, n_seeds=N_SEEDS, ccube_seeds=CCUBE_SEEDS, analysis=ANALYSIS, settings=settings)


def tumours(plan: dict) -> list[dict]:
    """One row per tumour: setting fields plus id, seed and replicate number."""
    rows = []
    for s in plan['settings']:
        for j in range(plan['n_seeds']):
            idx = s['index'] * plan['n_seeds'] + j
            rows.append(dict(s, id=f'sv{idx:04d}', seed=plan['seed_base'] + idx, rep=j))
    return rows


def simulate_truth(t: dict):
    """Mirror of power_scale.simulate that also returns the true subclone mask; asserted equal to it in `check`."""
    rng = np.random.default_rng([5151, t['seed']])
    n = t['n_snv']
    if t['generator'] == 'ho':
        depth = np.maximum(rng.negative_binomial(NB_SIZE, NB_SIZE / (NB_SIZE + t['depth']), n), 8)
    else:
        depth = np.maximum(rng.poisson(t['depth'], n), 8)
    gain = rng.random(n) < GAIN_FRAC[t['generator']]
    major, minor = np.where(gain, 2, 1), np.ones(n, int)
    rng.random(n)  # multiplicity draw (mult2 = 0), consumed to keep the stream aligned with power_scale
    sub = rng.random(n) < t['frac_sub']
    ccf = np.where(sub, 1.0 - t['delta'], 1.0) if t['delta'] else np.ones(n)
    q = np.minimum(ccf * t['purity'] / (t['purity'] * (major + minor) + 2 * (1 - t['purity'])), 0.95)
    if t['rho'] > 0:
        if t['generator'] == 'ho':
            q = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + rng.normal(0, np.sqrt(t['rho'] / (q * (1 - q)))))))
        else:
            k = 1 / t['rho'] - 1
            q = rng.beta(q * k, (1 - q) * k)
    return rng.binomial(depth, q), depth, major, minor, sub & (t['delta'] > 0)


def check(t: dict) -> None:
    from power_scale import simulate
    ref = simulate(t['seed'], t['depth'], t['purity'], t['rho'], t['delta'], generator=t['generator'], frac_sub=t['frac_sub'],
                   n_snv=t['n_snv'], mult2=0.0, gain_frac=GAIN_FRAC[t['generator']])
    mine = simulate_truth(t)
    assert all(np.array_equal(a, b) for a, b in zip(ref, mine[:4])), f"simulate_truth differs from power_scale.simulate for {t['id']}"


def export_tables(t: dict) -> None:
    alt, depth, major, minor, _ = simulate_truth(t)
    ids = [f's{i}' for i in range(len(alt))]
    pd.DataFrame(dict(mutation_id=ids, sample_id='T', ref_counts=depth - alt, alt_counts=alt, major_cn=major, minor_cn=minor,
                      normal_cn=2, tumour_content=t['purity'])).to_csv(DATA / 'pyclone' / f"{t['id']}.tsv", sep='\t', index=False)
    if t['rep'] < CCUBE_SEEDS:
        pd.DataFrame(dict(mutation_id=ids, var_counts=alt, ref_counts=depth - alt, total_counts=depth, normal_cn=2, purity=t['purity'],
                          major_cn_sub1=major, minor_cn_sub1=minor, total_cn_sub1=major + minor, frac_cn_sub1=1.0, major_cn_sub2=major,
                          minor_cn_sub2=minor, total_cn_sub2=major + minor, frac_cn_sub2=0.0, subclonal_cn=False)
                     ).to_csv(DATA / 'ccube' / f"{t['id']}.tsv", sep='\t', index=False)


def done_ids(path: Path) -> set:
    return {json.loads(l)['id'] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()


def run_v3() -> None:
    import partition_model_v3 as model
    out = RESULTS / 'screen_validation_v3.jsonl'
    done = done_ids(out)
    for t in (x for x in tumours(json.loads(PLAN.read_text())) if x['id'] not in done):
        alt, depth, major, minor, _ = simulate_truth(t)
        res = model.fit_predict(alt, depth, major, minor, t['purity'], seed=t['seed'])
        labels = np.asarray(res['labels'], int)
        rec = dict(id=t['id'], status='ok', k=int(res['k']), labels=labels.tolist(),
                   cp=[round(float(x), 5) for x in np.asarray(res['centers'], float)[labels]])
        with out.open('a') as f:
            f.write(json.dumps(rec) + '\n')
        print(t['id'], rec['k'], flush=True)


def run_pyclone(shard: int, n_shards: int) -> None:
    from run_accuracy_reruns import pyclone
    out = RESULTS / f'screen_validation_pyclone-binomial_{shard}of{n_shards}.jsonl'
    done = done_ids(out)
    for i, t in enumerate(tumours(json.loads(PLAN.read_text()))):
        if i % n_shards != shard or t['id'] in done:
            continue
        path, rec = DATA / 'pyclone' / f"{t['id']}.tsv", dict(id=t['id'])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                labels, cp, k = pyclone(path, 'binomial', Path(tmp), t['purity'], [f's{i}' for i in range(t['n_snv'])])
            rec.update(status='ok', k=int(k), labels=labels.tolist(), cp=[round(float(x), 5) for x in cp])
        except Exception as error:  # noqa: BLE001 - recorded per tumour, the run continues
            rec.update(status=f'{type(error).__name__}: {str(error)[:200]}', k=None)
        with out.open('a') as f:
            f.write(json.dumps(rec) + '\n')
        print(t['id'], rec['status'], rec.get('k'), flush=True)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'plan':
        if PLAN.exists():
            sys.exit(f'{PLAN} exists; the plan is written once, before any run')
        DATA.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(make_plan(), indent=1))
        print('wrote', PLAN)
    elif cmd == 'export':
        for sub in ('pyclone', 'ccube'):
            (DATA / sub).mkdir(parents=True, exist_ok=True)
        rows = tumours(json.loads(PLAN.read_text()))
        for t in rows[:6]:
            check(t)
        for t in rows:
            export_tables(t)
        print('exported', len(rows), 'tumours')
    elif cmd == 'run-v3':
        run_v3()
    elif cmd == 'run-pyclone':
        run_pyclone(int(sys.argv[2]), int(sys.argv[3]))
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
