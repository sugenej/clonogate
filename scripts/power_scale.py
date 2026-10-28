"""Scaled, resumable detection-power study. Usage: power_scale.py MODEL(v3|v32) NSEEDS OUT.jsonl BUDGET_SECONDS
MODEL v3 runs the rho == 0 cells (binomial model); v32 runs rho > 0 cells with model rho = sim rho (oracle match).
Use ONE model per process (v3.2 patches v3 on import).  Appends one JSON line per finished condition.
Cell sets: MAIN = depth {30,60,100} x purity {.4,.6,.8} x rho {0,.005,.02} x delta {0,.1,.2,.3,.4}, 700 SNVs, 50/50,
CN-gain 20% (m=1).  VARIANTS (depth 60, purity .6, rho {0,.02}, delta {0,.2,.3}): unequal clones (20% subclone),
300 / 1500 SNVs, multiplicity-2 gains (half of gain SNVs precede the gain), and a held-out generator HO (negative
binomial depth, logit-normal noise with matched variance, 35% CN-gain)."""
import sys
import json
import time
import itertools
from pathlib import Path
import numpy as np

sys.path.append(str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'model'))
DEFAULT = dict(generator='bb', frac_sub=0.5, n_snv=700, mult2=0.0, gain_frac=0.2)
DEPTHS, PURITIES, RHOS, DELTAS = (30, 60, 100), (0.4, 0.6, 0.8), (0.0, 0.005, 0.02), (0.0, 0.1, 0.2, 0.3, 0.4)
VARIANTS = (dict(frac_sub=0.2), dict(n_snv=300), dict(n_snv=1500), dict(mult2=0.5),
            dict(generator='ho', gain_frac=0.35))
NB_SIZE = 8.0


def simulate(seed, depth_mean, purity, rho, delta, generator='bb', frac_sub=0.5, n_snv=700, mult2=0.0, gain_frac=0.2):
    rng = np.random.default_rng([5151, seed])
    if generator == 'ho':
        depth = np.maximum(rng.negative_binomial(NB_SIZE, NB_SIZE / (NB_SIZE + depth_mean), n_snv), 8)
    else:
        depth = np.maximum(rng.poisson(depth_mean, n_snv), 8)
    gain = rng.random(n_snv) < gain_frac
    major = np.where(gain, 2, 1)
    minor = np.ones(n_snv, int)
    mult = np.where(gain & (rng.random(n_snv) < mult2), 2, 1)
    sub = rng.random(n_snv) < frac_sub
    ccf = np.where(sub, 1.0 - delta, 1.0) if delta else np.ones(n_snv)
    q = np.minimum(mult * ccf * purity / (purity * (major + minor) + 2 * (1 - purity)), 0.95)
    if rho > 0:
        if generator == 'ho':
            s = np.sqrt(rho / (q * (1 - q)))
            q = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + rng.normal(0, s))))
        else:
            k = 1 / rho - 1
            q = rng.beta(q * k, (1 - q) * k)
    return rng.binomial(depth, q), depth, major, minor


def cells():
    for d, p, r, dl in itertools.product(DEPTHS, PURITIES, RHOS, DELTAS):
        yield dict(DEFAULT, depth=d, purity=p, rho=r, delta=dl)
    for var in VARIANTS:
        for r, dl in itertools.product((0.0, 0.02), (0.0, 0.2, 0.3)):
            yield dict(DEFAULT, **var, depth=60, purity=0.6, rho=r, delta=dl)


def cell_key(which, c):
    return (f"{which}|{c['generator']}|d{c['depth']}|p{c['purity']}|r{c['rho']}|dl{c['delta']}|fs{c['frac_sub']}"
            f"|n{c['n_snv']}|m{c['mult2']}")


def main() -> None:
    which, n_seeds, out, budget = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])
    import importlib
    mod = importlib.import_module('partition_model_v3' if which == 'v3' else 'partition_model_v3_2')
    done = set()
    if out.exists():
        done = {json.loads(l)['key'] for l in out.read_text().splitlines() if l.strip()}
    t0 = time.time()
    for c in cells():
        if (c['rho'] == 0.0) != (which == 'v3'):
            continue
        key = cell_key(which, c)
        if key in done:
            continue
        if time.time() - t0 > budget:
            print('budget reached'); return
        if which == 'v32':
            mod.RHO = c['rho']
        sim_args = {k: v for k, v in c.items() if k not in ('depth', 'purity', 'rho', 'delta')}
        ks = [mod.fit_predict(*simulate(s, c['depth'], c['purity'], c['rho'], c['delta'], **sim_args),
                              c['purity'], seed=s)['k'] for s in range(n_seeds)]
        rec = dict(key=key, model=which, n=n_seeds, hits=int(sum(k >= 2 for k in ks)), ks=ks, **c)
        with out.open('a') as f:
            f.write(json.dumps(rec) + '\n')
        print(key, rec['hits'], '/', n_seeds, flush=True)
    print('ALL DONE')


if __name__ == '__main__':
    main()
