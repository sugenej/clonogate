"""Shared original-data loader; preserve both Battenberg states for CCube."""
from pathlib import Path
import pandas as pd
from run_smchet import load_inputs

out=Path('data/smchet/ccube_inputs')
out.mkdir(exist_ok=True)
for vcf in sorted(Path('data/smchet/vcfs').glob('mutect_*_noXY.vcf')):
    sample=vcf.name.removeprefix('mutect_').removesuffix('_noXY.vcf')
    tumor,_,depth=sample.split('.')
    base=Path('data/smchet/battenberg')/(tumor+'-'+depth+'_refit')
    if not Path(str(base)+'_subclones_noXY.txt').exists(): base=Path(str(base).removesuffix('_refit'))
    if not Path(str(base)+'_subclones_noXY.txt').exists(): continue
    d,f,purity,audit=load_inputs(vcf,Path(str(base)+'_subclones_noXY.txt'),Path(str(base)+'_cellularity_ploidy.txt'))
    f=f.reset_index(drop=True)
    table=pd.DataFrame(dict(mutation_id=d.mutation_id,var_counts=d.alt_count,
        ref_counts=d.ref_count,total_counts=d.alt_count+d.ref_count,normal_cn=2,purity=purity,
        major_cn_sub1=f.major_copy1,minor_cn_sub1=f.minor_copy1,
        total_cn_sub1=f.major_copy1+f.minor_copy1,frac_cn_sub1=f.state1,
        major_cn_sub2=f.major_copy2,minor_cn_sub2=f.minor_copy2,
        total_cn_sub2=f.major_copy2+f.minor_copy2,frac_cn_sub2=1-f.state1,
        subclonal_cn=f.state1<1))
    table.to_csv(out/(sample+'.tsv'),sep='\t',index=False)
