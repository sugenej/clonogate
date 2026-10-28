"""Recover FastClone designs that run_full_benchmark.py discarded for undefined assignment probabilities.

On the five failed designs FastClone clusters every mutation except 1-4 whose likelihood is -inf under every cluster, so its
posterior row is NaN (exp(-inf - -inf)).  The original runner rejected the whole design.  Recovery rule, fixed before scoring:
a mutation with a non-finite posterior row is assigned to the largest cluster (most finite-row argmax assignments); everything
else is FastClone's own output, and the count of such mutations is recorded in run.json.  Any other invalid output still fails.
Same cached-likelihood FastClone and 1-thread setting as the original runner.  Writes to reports/baseline_recovery_v1/full_frozen
so the originals stay untouched.  Usage: recover_fastclone.py SAMPLE [SAMPLE...]
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.append(str(Path(__file__).resolve().parent))
from run_smchet import load_inputs  # noqa: E402
from run_fastclone_cached import cached_weight_function  # noqa: E402
from run_ledger import atomic_json  # noqa: E402
import fastclone.subclone as fastclone  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'baseline_recovery_v1' / 'full_frozen'


def recover(sample: str) -> dict:
    """Run FastClone on one design and write the scorer's 1A/1B/1C/2A files; returns the audit record."""
    folder = OUT / sample / 'fastclone_core'
    folder.mkdir(parents=True, exist_ok=True)
    base = ROOT / 'data' / 'smchet_full' / sample
    inputs = [base / f'{sample}.mutect.vcf', base / f'{sample}.battenberg.txt', base / f'{sample}.cellularity_ploidy.txt']
    audit, start = dict(sample=sample, method='fastclone_core', recovery='nan_rows_to_largest_cluster'), time.perf_counter()
    try:
        _, f, purity, metadata = load_inputs(*inputs)
        audit.update(metadata)
        with threadpool_limits(limits=1):
            centers, probabilities = fastclone.infer_single_sample(f.copy(), purity)
        probabilities, centers = probabilities.to_numpy(), np.asarray(centers)
        finite = np.isfinite(probabilities).all(axis=1)
        if not np.allclose(probabilities[finite].sum(axis=1), 1):
            raise ValueError('Invalid assignment probabilities beyond undefined rows')
        if not np.isfinite(centers).all() or (centers < 0).any() or (centers > 1).any():
            raise ValueError('Invalid cellular prevalence')
        assigned = np.full(len(probabilities), -1)
        assigned[finite] = probabilities[finite].argmax(axis=1)
        assigned[~finite] = np.bincount(assigned[finite], minlength=probabilities.shape[1]).argmax()
        used, labels = np.unique(assigned, return_inverse=True)
        centers = centers[used]
        (folder / '1A.txt').write_text(str(purity) + '\n')
        (folder / '1B.txt').write_text(str(len(centers)) + '\n')
        np.savetxt(folder / '2A.txt', labels + 1, fmt='%d')
        np.savetxt(folder / '1C.txt', np.c_[np.arange(len(centers)) + 1, np.bincount(labels), centers], delimiter='\t',
                   fmt=['%d', '%d', '%.9f'])
        audit.update(status='ok', inferred_k=len(centers), n_mutations=len(probabilities), undefined_rows=int((~finite).sum()))
    except Exception:
        audit.update(status='error', error=traceback.format_exc())
    audit['seconds'] = time.perf_counter() - start
    atomic_json(folder / 'run.json', audit)
    return audit


if __name__ == '__main__':
    fastclone._estimate_subclone_weights, _ = cached_weight_function(fastclone._estimate_subclone_weights)
    for name in sys.argv[1:]:
        record = recover(name)
        print(name, record['status'], record.get('undefined_rows'), record.get('inferred_k'), round(record['seconds'], 1), flush=True)
