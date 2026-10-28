"""Export power-study simulations in CCube and PyClone-VI input formats. Usage: export_sim_baselines.py NSEEDS [OUT_DIR]
Grid: depth {30,100} x purity {.4,.8} x rho {0,.02} x delta {0,.25,.55}; same simulate() as power_study.py, seeds 0..N-1.
Writes OUT_DIR/ccube/ID.tsv, OUT_DIR/pyclone/ID.tsv and OUT_DIR/manifest.jsonl (true_k = 1 if delta == 0 else 2)."""
import sys
import json
import itertools
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from power_study import simulate  # noqa: E402

DEPTHS, PURITIES, RHOS, DELTAS = (30, 100), (0.4, 0.8), (0.0, 0.02), (0.0, 0.25, 0.55)


def cell_id(depth, purity, rho, delta, seed):
    return f'd{depth}_p{purity}_r{rho}_dl{delta}_s{seed}'


def ccube_table(alt, depth, major, minor, purity):
    n = len(alt)
    return pd.DataFrame(dict(
        mutation_id=[f's{i}' for i in range(n)], var_counts=alt, ref_counts=depth - alt, total_counts=depth,
        normal_cn=2, purity=purity, major_cn_sub1=major, minor_cn_sub1=minor, total_cn_sub1=major + minor,
        frac_cn_sub1=1.0, major_cn_sub2=major, minor_cn_sub2=minor, total_cn_sub2=major + minor, frac_cn_sub2=0.0,
        subclonal_cn=False))


def pyclone_table(alt, depth, major, minor, purity):
    n = len(alt)
    return pd.DataFrame(dict(
        mutation_id=[f's{i}' for i in range(n)], sample_id='T', ref_counts=depth - alt, alt_counts=alt,
        major_cn=major, minor_cn=minor, normal_cn=2, tumour_content=purity))


def main() -> None:
    n_seeds = int(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('data/sim_grid')
    (out / 'ccube').mkdir(parents=True, exist_ok=True)
    (out / 'pyclone').mkdir(parents=True, exist_ok=True)
    with (out / 'manifest.jsonl').open('w') as man:
        for depth, pur, rho, delta, seed in itertools.product(DEPTHS, PURITIES, RHOS, DELTAS, range(n_seeds)):
            alt, dep, major, minor = simulate(seed, depth, pur, rho, delta)
            cid = cell_id(depth, pur, rho, delta, seed)
            ccube_table(alt, dep, major, minor, pur).to_csv(out / 'ccube' / f'{cid}.tsv', sep='\t', index=False)
            pyclone_table(alt, dep, major, minor, pur).to_csv(out / 'pyclone' / f'{cid}.tsv', sep='\t', index=False)
            man.write(json.dumps(dict(id=cid, depth=depth, purity=pur, rho=rho, delta=delta, seed=seed,
                                      true_k=1 if delta == 0 else 2)) + '\n')
    print('wrote', n_seeds * len(DEPTHS) * len(PURITIES) * len(RHOS) * len(DELTAS), 'simulations to', out)


if __name__ == '__main__':
    main()
