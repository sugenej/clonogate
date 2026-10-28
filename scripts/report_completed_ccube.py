"""Post-score verifier for a retained snapshot; never reruns inference or scoring."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from run_ledger import atomic_json

METHODS={'full_frozen':['adaptive_dispersion','fastclone_core'],
    'full_ccube':['ccube_public'],'full_oi':['object_integration_6184761'],'full_phylogic':['phylogic_public']}


def score_paths(snapshot,root,experiment,score):
    sample,task,method=score['sample'],score['challenge'],score['method']
    return [snapshot/'reports'/experiment/sample/method/(task+'.txt'),
        root/'data/smchet_full'/sample/(sample+'.truth.'+task+'.txt'),
        root/'data/smchet_full'/sample/(sample+'.truth.scoring_vcf.vcf')]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];snapshot=root/'reports/full_ccube_evaluation_v1'
    originals=json.loads((snapshot/'snapshot.json').read_text())['original_sha256']
    samples=[r['sample'] for r in json.loads((snapshot/'reports/full_alignment.json').read_text())]
    ccube=json.loads((snapshot/'reports/full_ccube/official_scores.json').read_text())
    expected={(s,'ccube_public',t) for s in samples for t in ['1A','1B','1C','2A']}
    if len(ccube)!=len(expected) or {(r['sample'],r['method'],r['challenge']) for r in ccube}!=expected:
        raise ValueError('Snapshot scoring incomplete')
    final=snapshot/'evaluation_verified_v1';final.mkdir(exist_ok=False)
    # Point-output copies and terminal statuses must match snapshot originals.
    changed_by_scoring={'reports\\full_ccube\\official_scores.json'}
    for p,h in originals.items():
        if p not in changed_by_scoring and digest(snapshot/p)!=h:raise ValueError('Snapshot artifact changed: '+p)
    checks=[]
    for experiment in METHODS:
        path=snapshot/'reports'/experiment/'official_scores.json'
        if not path.exists():continue
        for row in json.loads(path.read_text()):
            if row['status']!='scored':continue
            paths=score_paths(snapshot,root,experiment,row)
            fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()
            if fingerprint!=row['input_sha256']:raise ValueError('Score fingerprint mismatch')
            checks.append(dict(sample=row['sample'],method=row['method'],challenge=row['challenge'],
                input_sha256=fingerprint,truth_sha256=digest(paths[1]),vcf_sha256=digest(paths[2])))
    command=[sys.executable,str(root/'scripts/summarize_full_benchmark.py')]
    subprocess.run(command,cwd=snapshot,check=True)
    report=(snapshot/'reports/FULL_BENCHMARK_RESULTS.md').read_text().replace('New (46)','Outside initial development (46)')
    report=report.replace('Partial runs are not final rankings or evidence of SOTA.',
        'CCube has51 terminal attempts; other baseline batches may be partial. These designs are all '
        'inspected now, not a fresh holdout. The46-design subgroup is outside the initial development '
        'overlap. This snapshot is not a final ranking or evidence of SOTA.')
    report+='\n\nSnapshot notes: original scoring completed, but its first post-score verifier used a '
    report+='truth filename without the .txt suffix and failed. The failed driver/logs remain retained. '
    report+='This separate verifier repairs the path handling without rerunning inference or scoring. '
    report+='It verifies every scored fingerprint and preserves all original report/prediction files.\n'
    (final/'REPORT.md').write_text(report)
    (final/'full_benchmark_scores.csv').write_bytes((snapshot/'reports/full_benchmark_scores.csv').read_bytes())
    (final/'report_completed_ccube.py').write_bytes(Path(__file__).read_bytes())
    original_changed=[p for p,h in originals.items() if digest(root/p)!=h]
    if original_changed:raise ValueError('Original input changed: '+str(original_changed))
    preserved=json.loads((root/'reports/geometry_calibration_v1/preservation_audit.json').read_text())['frozen_prediction_sha256']
    if any(digest(root/p)!=h for p,h in preserved.items()):raise ValueError('Frozen prediction changed')
    freeze=json.loads((root/'research/FULL_BENCHMARK_FREEZE.json').read_text())['source_sha256']
    if any(digest(root/p)!=h for p,h in freeze.items()):raise ValueError('Frozen model changed')
    atomic_json(final/'verification.json',dict(original_files_checked=len(originals),
        score_checks=checks,frozen_predictions_checked=len(preserved),frozen_sources_checked=len(freeze),
        mismatches=[],report_sha256=digest(final/'REPORT.md'),
        scorer_sha256=digest(root/'scripts/score_smchet.py'),summary_sha256=digest(root/'scripts/summarize_full_benchmark.py'),
        verifier_sha256=digest(Path(__file__))))
    print(final/'REPORT.md',flush=True)


if __name__=='__main__':main()
