"""Resumable detection-power study. Usage: power_study.py MODEL(v3|v32|v32c|v36) NSEEDS OUT.jsonl BUDGET_SECONDS
v32c = v3.2 at fixed rho 0.02 for all sim rho; v36 = v3.6 (rho estimated from the data).
Conditions: depth x purity x rho x (1 clone | 2 clones at CCF 1.0 and 1.0-delta, 50/50 SNVs). 700 SNVs, 20% CN-gain (m=1).
v32 uses model rho = sim rho (matched; sim rho 0 skipped). Appends one JSON line per finished condition."""
import sys
import json
import time
import itertools
from pathlib import Path
import numpy as np

sys.path.append(str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'model'))
N_SNV = 700
CEILING_RHO = 0.02   # v32c: v3.2 with this fixed rho for every sim rho (rho-agnostic "ceiling" baseline)
DEPTHS, PURITIES, RHOS, DELTAS = (30, 100), (0.4, 0.8), (0.0, 0.005, 0.02), (0.1, 0.25, 0.4, 0.55)


def simulate(seed, depth_mean, purity, rho, delta):
    rng = np.random.default_rng([4242, seed])
    depth = np.maximum(rng.poisson(depth_mean, N_SNV), 8)
    major = np.where(rng.random(N_SNV) < 0.2, 2, 1)
    minor = np.ones(N_SNV, int)
    ccf = np.where(rng.random(N_SNV) < 0.5, 1.0, 1.0 - delta) if delta else np.ones(N_SNV)
    q = ccf * purity / (purity * (major + minor) + 2 * (1 - purity))
    if rho > 0:
        s = 1 / rho - 1
        q = rng.beta(q * s, (1 - q) * s)
    return rng.binomial(depth, q), depth, major, minor


def main() -> None:
    which, n_seeds, out, budget = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])
    import importlib
    mod = importlib.import_module({'v3': 'partition_model_v3', 'v36': 'partition_model_v3_6'}.get(which, 'partition_model_v3_2'))
    done = set()
    if out.exists():
        done = {json.loads(l)['key'] for l in out.read_text().splitlines() if l.strip()}
    t0 = time.time()
    for depth, pur, rho, delta in itertools.product(DEPTHS, PURITIES, RHOS, (0.0,) + DELTAS):
        key = f'{which}|d{depth}|p{pur}|r{rho}|dl{delta}'
        if key in done or (which == 'v32' and rho == 0):
            continue
        if time.time() - t0 > budget:
            print('budget reached'); return
        if which == 'v32':
            mod.RHO = rho
        elif which == 'v32c':
            mod.RHO = CEILING_RHO
        ks = [mod.fit_predict(*simulate(s, depth, pur, rho, delta), pur, seed=s)['k'] for s in range(n_seeds)]
        rec = dict(key=key, model=which, depth=depth, purity=pur, rho=rho, delta=delta, n=n_seeds,
                   hits=int(sum(k >= 2 for k in ks)), ks=ks)
        with out.open('a') as f:
            f.write(json.dumps(rec) + '\n')
        print(key, rec['hits'], '/', n_seeds, flush=True)
    print('ALL DONE')


if __name__ == '__main__':
    main()
