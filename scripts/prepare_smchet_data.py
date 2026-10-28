"""Inventory an original SMC-Het archive and safely extract compact truth files."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def prepare(archive, destination, all_small=False, benchmark_inputs=False):
    destination=destination.resolve()
    destination.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    records=[]
    suffixes=('.1A.txt','.1B.txt','.1C.txt','.2A.txt','.3A.txt','.scoring_vcf.vcf')
    if benchmark_inputs:
        suffixes+=('.mutect.vcf','.battenberg.txt','.cellularity_ploidy.txt')
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            relative=PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in member.name:
                raise ValueError('Unsafe archive member path')
            selected=member.isfile() and (all_small or member.name.endswith(suffixes))
            record={'name':member.name,'bytes':member.size,'extracted':selected}
            if selected:
                if member.size>50*1024*1024:raise ValueError('Unexpectedly large compact truth file')
                target=destination.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(destination):raise ValueError('Archive path escapes destination')
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():raise FileExistsError(target)
                with tf.extractfile(member) as source,target.open('wb') as out:shutil.copyfileobj(source,out)
            records.append(record)
            if len(records)%100==0:print(f'Indexed {len(records)} members',flush=True)
    manifest={'archive':str(archive.resolve()),'archive_bytes':archive.stat().st_size,
              'sha256':digest.hexdigest(),'destination':str(destination),'members':records}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'members':len(records),'extracted':sum(r['extracted'] for r in records),
                      'sha256':digest.hexdigest()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--all-small',action='store_true',help='Extract all regular members, each limited to 50 MiB')
    p.add_argument('--benchmark-inputs',action='store_true',help='Also extract raw MuTect VCF and Battenberg inputs, excluding quadratic matrices')
    a=p.parse_args();prepare(a.archive,a.destination,a.all_small,a.benchmark_inputs)
