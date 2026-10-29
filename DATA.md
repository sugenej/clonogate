# Data sources

No raw benchmark data are stored in this repository. This file lists what each analysis needs and where it comes from.
The license in `LICENSE` covers the code only, not the datasets below.

## SMC-Het benchmark (simulated tumors)
- ICGC-TCGA DREAM Somatic Mutation Calling Tumour Heterogeneity Challenge: 51 simulated tumors with known truth
  (Salcedo et al., 2025; full reference in the report). Obtain the data and the official scorer from the challenge
  distribution; they are not re-hosted here because they carry their own access terms.
- Used by: `scripts/prepare_smchet_data.py` (extracts the truth files and benchmark inputs from the original archive),
  `prepare_ccube*.py`, `prepare_full_baselines.py`, `run_baseline_recovery.sh`, `make_figures*.py`.
- Scores are computed with the official scorer, unmodified (`scripts/check_original_scorer.py`).

## Genome in a Bottle HG002 / HG003 (real reads, Section 8.2)
- Truth: GRCh38 v4.2.1 benchmark VCFs and `_noinconsistent.bed` for HG002 and HG003 (Zook et al., 2016; Wagner et al., 2022),
  from the GIAB resource at https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/.
- Reads: Illumina, about 60x per sample, piled up with `samtools mpileup` over chr20:1-30 Mb.
  The exact read-file accessions are not yet recorded here; add them before claiming reproducibility.
- Expected layout (`GIAB_DIR`, default `data/giab/`): `HG00[23]_GRCh38_1_22_v4.2.1_benchmark.vcf.gz`, the matching
  `_noinconsistent.bed` files, `pileup_HG002.txt`, `pileup_HG003.txt`.
- Included here: `reports/giab_mixture_check/counts.tsv.gz`, per-site read counts derived from the above
  (output of `giab_mixture_check.py counts`), plus the run outputs `runs.jsonl` and `runs_v32.jsonl`. With these,
  `giab_mixture_check.py report`, `heavy_tail_check.py` and `make_figure_giab.py` should run without the raw reads.
  This has not been verified from a clean checkout.

## SMaHT HapMap6 mixture (Section 8.1, no result reported)
- Kong et al., 2025 (preprint) and the SMaHT data portal, https://data.smaht.org.
- The mixture reads were not available to us. `scripts/hapmap_check.py` is included for completeness; it expects
  `sites.tsv` and `pileup.txt` in `HAPMAP_DIR` (default `data/hm/`). Its helper functions are also imported by the GIAB scripts.

## Published tumor datasets assessed but not used
Wang et al. 2014, Gawad et al. 2014, Leung et al. 2017, Morita et al. 2020, SEQC2 HCC1395: assessed in the report as
candidates and found unsuitable for single-sample validation with known subclones. No data from them are used or included.
