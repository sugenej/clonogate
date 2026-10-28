"""Extra detection-power cells that vary subclone fraction and SNV count (direction 2).  Usage: power_fracn.py MODEL(v3|v32) NSEEDS OUT.jsonl BUDGET_SECONDS
Same simulator, models and record layout as power_scale.py (resumable, one model per process; v3 runs rho == 0, v32 runs rho > 0 with
model rho = sim rho).  Cells: depth 60, purity .6, subclone fraction {.1,.2,.35,.5} x SNV count {300,700,1500} x delta {.15,.25,.35}
x rho {0,.02}; 50/50 is fs .5.  Fills the gap that left the subclone-fraction term unidentified in BOUNDARY_FIT.md."""
import importlib
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import power_scale as ps  # noqa: E402

FRACTIONS = (0.1, 0.2, 0.35, 0.5)
N_SNVS = (300, 700, 1500)
DELTAS = (0.15, 0.25, 0.35)
RHOS = (0.0, 0.02)
DEPTH, PURITY = 60, 0.6


def cells():
    for fs, n, dl, r in itertools.product(FRACTIONS, N_SNVS, DELTAS, RHOS):
        yield dict(ps.DEFAULT, frac_sub=fs, n_snv=n, depth=DEPTH, purity=PURITY, rho=r, delta=dl)


def main() -> None:
    which, n_seeds, out, budget = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])
    if which not in ('v3', 'v32'):
        raise SystemExit('MODEL must be v3 or v32')
    mod = importlib.import_module('partition_model_v3' if which == 'v3' else 'partition_model_v3_2')
    done = {json.loads(l)['key'] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t0 = time.time()
    for c in cells():
        if (c['rho'] == 0.0) != (which == 'v3'):
            continue
        key = ps.cell_key(which, c)
        if key in done:
            continue
        if time.time() - t0 > budget:
            print('budget reached')
            return
        if which == 'v32':
            mod.RHO = c['rho']
        sim_args = {k: v for k, v in c.items() if k not in ('depth', 'purity', 'rho', 'delta')}
        ks = [mod.fit_predict(*ps.simulate(s, c['depth'], c['purity'], c['rho'], c['delta'], **sim_args),
                              c['purity'], seed=s)['k'] for s in range(n_seeds)]
        rec = dict(key=key, model=which, n=n_seeds, hits=int(sum(k >= 2 for k in ks)), ks=ks, **c)
        with out.open('a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(rec) + '\n')
        print(key, rec['hits'], '/', n_seeds, flush=True)
    print('ALL DONE')


if __name__ == '__main__':
    main()
