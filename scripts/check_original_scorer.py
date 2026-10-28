"""Python 2 integration invariants using only the original scorer."""
from __future__ import print_function
import os
import sys
import tempfile
import json
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root, 'external/smc-scoring/smc_het_eval'))
import SMCScoring as s
s.INFO = False
folder = tempfile.mkdtemp(prefix='resolveclone-scorer-')
def write(name, data):
    p = os.path.join(folder, name)
    with open(p, 'w') as f:
        f.write(data)
    return p
vcf = write('scoring.vcf', '#CHROM\tPOS\n'+''.join('1\t%d\tTrue\n' % i for i in range(1,11)))
results = {}
for challenge, data in [('1A','0.6\n'), ('1B','2\n'), ('1C','1\t5\t0.6\n2\t5\t0.3\n'), ('2A','1\n'*5+'2\n'*5)]:
    p = write(challenge+'.txt', data)
    value = s.scoreChallenge(challenge, [p], [p], vcf)
    scalar = value[-1] if isinstance(value, list) else value
    assert abs(scalar-1) < 1e-10, (challenge, value, s.err_msgs)
    results[challenge] = value
with open(os.path.join(root, 'reports/smchet_scorer_invariants.json'), 'w') as f:
    json.dump(results, f, indent=2)
print('Perfect prediction invariants passed:', results)
