"""Two-clone detection ladder. Usage: power_ladder.py MODULE NSEEDS OUT.json
50/50 SNVs at CCF 1.0 and CCF2 in (0.8,0.6,0.4); beta-binomial rho in (0,.005,.02). Reports fraction with fitted k>=2.
For partition_model_v3_2 the model rho is set equal to the sim rho (skips rho=0)."""
import sys
import json
import importlib
import numpy as np

sys.path.append(str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'model'))
N_SNV, PURITY = 700, 0.7


def simulate(seed: int, ccf2: float, rho: float):
    rng = np.random.default_rng([888, seed])
    depth = np.maximum(rng.poisson(60, N_SNV), 10)
    major = np.where(rng.random(N_SNV) < 0.2, 2, 1)
    minor = np.ones(N_SNV, int)
    ccf = np.where(rng.random(N_SNV) < 0.5, 1.0, ccf2)
    q = ccf * PURITY / (PURITY * (major + minor) + 2 * (1 - PURITY))
    if rho > 0:
        s = 1 / rho - 1
        q = rng.beta(q * s, (1 - q) * s)
    return rng.binomial(depth, q), depth, major, minor


def main() -> None:
    mod = importlib.import_module(sys.argv[1])
    n_seeds, out = int(sys.argv[2]), {}
    for rho in (0.0, 0.005, 0.02):
        if sys.argv[1].endswith('v3_2') and rho == 0:
            continue
        if hasattr(mod, 'RHO'):
            mod.RHO = rho
        for ccf2 in (0.8, 0.6, 0.4):
            ks = [mod.fit_predict(*simulate(s, ccf2, rho), PURITY, seed=s)['k'] for s in range(n_seeds)]
            out[f'rho{rho}_ccf2_{ccf2}'] = dict(detected=int(sum(k >= 2 for k in ks)), n=n_seeds, ks=ks)
            print(sys.argv[1], 'rho', rho, 'ccf2', ccf2, 'detected', sum(k >= 2 for k in ks), '/', n_seeds, flush=True)
    json.dump(out, open(sys.argv[3], 'w'))


if __name__ == '__main__':
    main()
