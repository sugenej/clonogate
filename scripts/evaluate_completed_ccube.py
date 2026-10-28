"""Snapshot terminal CCube outputs, score copies, summarize without changing originals."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from run_ledger import atomic_json


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    destination=root/'reports/full_ccube_evaluation_v1';destination.mkdir(exist_ok=False)
    alignment=json.loads((root/'reports/full_alignment.json').read_text())
    samples=[r['sample'] for r in alignment]
    if len(samples)!=51 or len(set(samples))!=51:raise ValueError('Expected51 unique designs')
    ccube=root/'reports/full_ccube'
    if any(not (ccube/s/'ccube_public/status').is_file() for s in samples):raise ValueError('CCube batch incomplete')
    original={}
    def capture(path):
        relative=path.relative_to(root);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        content=path.read_bytes();target.write_bytes(content);original[str(relative)]=hashlib.sha256(content).hexdigest()
    methods={'full_frozen':['adaptive_dispersion','fastclone_core'],
        'full_ccube':['ccube_public'],'full_oi':['object_integration_6184761'],
        'full_phylogic':['phylogic_public']}
    for experiment,names in methods.items():
        base=root/'reports'/experiment
        for name in ['official_scores.json','manifest.json','runs.csv']:
            if (base/name).is_file():capture(base/name)
        for sample in samples:
            for method in names:
                folder=base/sample/method
                # Only terminal Phylogic cases are copied; live partial outputs are not read.
                if experiment=='full_phylogic' and not (folder/'status').exists():continue
                for name in ['1A.txt','1B.txt','1C.txt','2A.txt','status','run.json','export.json','exit_code']:
                    if (folder/name).is_file():capture(folder/name)
    for name in ['reports/full_alignment.json','reports/FULL_BENCHMARK_RESULTS.md',
        'reports/PUBLISHED_CLONE_COUNTS.md','scripts/score_smchet.py','scripts/summarize_full_benchmark.py',
        'scripts/evaluate_completed_ccube.py','research/FULL_BENCHMARK_FREEZE.json']:
        capture(root/name)
    atomic_json(destination/'snapshot.json',dict(original_sha256=original,
        designation='Completed CCube plus partial other baseline snapshot; all designs inspected',
        original_report_sha256=original['reports\\FULL_BENCHMARK_RESULTS.md']))
    # Absolute data path is translated explicitly; scorer arguments remain structured.
    def linux(path):
        path=path.resolve();return '/mnt/'+path.drive[0].lower()+str(path)[2:].replace('\\','/')
    command=['wsl.exe','-e','/home/sugen/resolveclone-runtime/py27/bin/python',
        linux(root/'scripts/score_smchet.py'),linux(destination/'reports/full_ccube'),
        'ccube_public',linux(root/'data/smchet_full')]
    with (destination/'scoring.log').open('w') as log:subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic_json(destination/'scoring_command.json',command)
    # Verify every displayed scored cache against actual retained prediction/truth/VCF bytes.
    score_checks=0
    for experiment,names in methods.items():
        path=destination/'reports'/experiment/'official_scores.json'
        if not path.exists():continue
        for score in json.loads(path.read_text()):
            if score['status']!='scored':continue
            sample=score['sample'];task=score['challenge'];method=score['method']
            pred=destination/'reports'/experiment/sample/method/(task+'.txt')
            truth=root/'data/smchet_full'/sample/(sample+'.truth.'+task)
            vcf=root/'data/smchet_full'/sample/(sample+'.truth.scoring_vcf.vcf')
            fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in [pred,truth,vcf])).hexdigest()
            if fingerprint!=score['input_sha256']:raise ValueError('Score cache mismatch')
            score_checks+=1
    subprocess.run([sys.executable,str(root/'scripts/summarize_full_benchmark.py')],cwd=destination,check=True)
    report=(destination/'reports/FULL_BENCHMARK_RESULTS.md').read_text()
    report=report.replace('New (46)','Outside initial development (46)')
    report=report.replace('Partial runs are not final rankings or evidence of SOTA.',
        'CCube has51 terminal attempts; other baselines may be partial. All these designs are now inspected, '
        'not a fresh holdout. The46-design subgroup is outside the initial development overlap. '
        'This snapshot is not a final ranking or evidence of SOTA.')
    (destination/'REPORT.md').write_text(report)
    for relative,h in original.items():
        # Summarization/scoring changes only copied report/cache files, never original inputs.
        if digest(root/relative)!=h:raise ValueError('Original snapshot input changed: '+relative)
    preserved=json.loads((root/'reports/geometry_calibration_v1/preservation_audit.json').read_text())['frozen_prediction_sha256']
    if any(digest(root/p)!=h for p,h in preserved.items()):raise ValueError('Frozen prediction changed')
    freeze=json.loads((root/'research/FULL_BENCHMARK_FREEZE.json').read_text())['source_sha256']
    if any(digest(root/p)!=h for p,h in freeze.items()):raise ValueError('Frozen model changed')
    atomic_json(destination/'verification.json',dict(original_files_checked=len(original),
        scored_fingerprints_checked=score_checks,frozen_predictions_checked=len(preserved),
        frozen_sources_checked=len(freeze),mismatches=[],
        report_sha256=digest(destination/'REPORT.md')))
    print(destination/'REPORT.md',flush=True)


if __name__=='__main__':main()
