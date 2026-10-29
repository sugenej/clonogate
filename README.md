# ClonoGate

Single-sample subclonal reconstruction with a significance gate on a copy-number-aware binomial mixture
(`partition_v3`, plus the beta-binomial variant `partition_v3_2`).

**Status: work in progress, shared before completion.** The accompanying report is
`reports/STS_REPORT_DRAFT_v9.md` (`.docx` also included).

What the evidence supports, and what it does not:
- Evaluated on the SMC-Het simulated benchmark designs, on simulated overdispersion tests, and on a
  computational mixture of public GIAB HG002/HG003 reads (real sequencing reads, germline sites standing in for clones).
- **No validation on real tumors with known clonal structure.** No claim of state-of-the-art performance.
- On real reads the gate still false-splits single-clone samples more often than its nominal rate; see report Section 8.2.

## Layout
- `model/` – the mixture models (`partition_model_v3.py` is ClonoGate; `partition_model_v3_2.py` is the beta-binomial variant).
- `scripts/` – experiment, baseline-recovery and figure scripts used in the report.
- `reports/` – the report, figures, and small result files (GIAB mixture check, heavy-tail check).

## Reproducing
Scripts add `model/` to the path relative to the repo. Input data are **not** included: SMC-Het benchmark files and
the GIAB reads must be obtained separately; see `DATA.md` for sources and expected layout (set `GIAB_DIR`, `HAPMAP_DIR`).
Environment pinning and a clean end-to-end run have not been done yet.

## License
MIT for the code (see `LICENSE`). Datasets keep their own terms; see `DATA.md`.
