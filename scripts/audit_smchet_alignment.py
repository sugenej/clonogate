"""Evaluation-only check; does not expose truth labels to inference."""
import json
from pathlib import Path
rows = []
def ids(path):
    return [tuple(l.split('\t')[i] for i in [0,1,3,4]) for l in path.read_text().splitlines() if not l.startswith('#')]
for vcf in sorted(Path('data/smchet/vcfs').glob('mutect_*_noXY.vcf')):
    sample = vcf.name.removeprefix('mutect_').removesuffix('_noXY.vcf')
    truth = Path('data/smchet/truth/single_caller')/('MuTect_'+sample+'.truth.scoring_vcf.vcf')
    row = dict(sample=sample, variants=len(ids(vcf)), identical_order=ids(vcf)==ids(truth))
    if not row['identical_order']:
        raise ValueError('VCF alignment mismatch: '+sample)
    rows.append(row)
Path('reports/smchet_alignment.json').write_text(json.dumps(rows, indent=2))
print('Identical inference/scoring variant order:', len(rows), 'samples')
