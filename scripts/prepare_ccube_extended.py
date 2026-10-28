"""Reuse successful compact outputs with provenance; retry only timeout jobs."""
import hashlib
import json
import shutil
from pathlib import Path

for source in Path('reports/ccube_public_v2').glob('*/ccube_public'):
    if (source/'status').read_text().strip() != 'ok': continue
    target=Path('reports/ccube_public_v3')/source.parent.name/'ccube_public'
    target.mkdir(parents=True,exist_ok=True)
    if (target/'status').exists(): continue
    hashes={}
    for name in ['1A.txt','1B.txt','1C.txt','2A.txt','exit_code']:
        shutil.copy2(source/name,target/name)
        hashes[name]=hashlib.sha256((source/name).read_bytes()).hexdigest()
    (target/'reused_run.json').write_text(json.dumps(dict(source=str(source),sha256=hashes),indent=2))
    shutil.copy2(source/'status',target/'status')
