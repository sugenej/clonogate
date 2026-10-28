"""Raw-count/CNA adapters for full benchmark; no truth file access."""
import json
from pathlib import Path
import pandas as pd
from run_smchet import load_inputs

root=Path('data/smchet_full')
assert (root/'manifest.json').exists(), 'Extraction must be complete'
ccube=root/'ccube_inputs'
phylogic=root/'phylogic_inputs'
ccube.mkdir(exist_ok=True)
phylogic.mkdir(exist_ok=True)
records=[]
for vcf in sorted(root.glob('*/*.mutect.vcf')):
    sample=vcf.name.removesuffix('.mutect.vcf')
    try:
        d,f,purity,audit=load_inputs(vcf,vcf.parent/(sample+'.battenberg.txt'),vcf.parent/(sample+'.cellularity_ploidy.txt'))
        f=f.reset_index(drop=True)
        table=pd.DataFrame(dict(mutation_id=d.mutation_id,var_counts=d.alt_count,
            ref_counts=d.ref_count,total_counts=d.alt_count+d.ref_count,normal_cn=2,purity=purity,
            major_cn_sub1=f.major_copy1,minor_cn_sub1=f.minor_copy1,
            total_cn_sub1=f.major_copy1+f.minor_copy1,frac_cn_sub1=f.state1,
            major_cn_sub2=f.major_copy2,minor_cn_sub2=f.minor_copy2,
            total_cn_sub2=f.major_copy2+f.minor_copy2,frac_cn_sub2=1-f.state1,
            subclonal_cn=f.state1<1))
        table.to_csv(ccube/(sample+'.tsv'),sep='\t',index=False)
        parts=d.mutation_id.str.split(':',expand=True)
        major=f.state1*f.major_copy1+(1-f.state1)*f.major_copy2
        minor=f.state1*f.minor_copy1+(1-f.state1)*f.minor_copy2
        maf=pd.DataFrame(dict(Chromosome=parts[0],Start_position=parts[1],Reference_Allele=parts[2],
            Tumor_Seq_Allele2=parts[3],t_alt_count=d.alt_count,t_ref_count=d.ref_count,
            local_cn_a1=minor,local_cn_a2=major))
        keep=major>0
        maf[keep].to_csv(phylogic/(sample+'.maf'),sep='\t',index=False)
        (phylogic/(sample+'.json')).write_text(json.dumps(dict(purity=purity,zero_cn_omitted=int((~keep).sum()),total_variants=len(d)),indent=2))
        records.append(dict(sample=sample,status='ok',**audit))
    except Exception as error:
        records.append(dict(sample=sample,status='error',error=repr(error)))
(root/'baseline_inputs.json').write_text(json.dumps(records,indent=2))
print('Prepared',sum(r['status']=='ok' for r in records),'of',len(records),'designs')
