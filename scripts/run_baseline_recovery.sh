#!/usr/bin/env bash
# Re-run one failed SMC-Het benchmark design for a published baseline into reports/baseline_recovery_v1 (originals untouched).
# Usage: run_baseline_recovery.sh TOOL SAMPLE LIMIT_SECONDS     TOOL = ccube | phylogic | phylogic_bracketed
# Same command lines as run_ccube.sh / run_phylogic.sh; phylogic_bracketed adds only the numerical root-bracket fallback.
# Status files use the originals' vocabulary (ok / timeout_<limit>s / error); an existing status file means the job is done.
set -u
tool="$1"; sample="$2"; limit="$3"
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$root/reports/baseline_recovery_v1/$tool/$sample"
mkdir -p "$work"
cd "$work"
if [ -e status ]; then echo "$tool $sample already $(cat status)"; exit 0; fi
code=0
case "$tool" in
  ccube)
    timeout "$limit" /home/sugen/resolveclone-runtime/r/bin/Rscript "$root/scripts/run_ccube.R" "$root/data/smchet_full/ccube_inputs/$sample.tsv" > run.log 2>&1 || code=$?
    ;;
  phylogic|phylogic_bracketed)
    input="$root/data/smchet_full/phylogic_inputs/$sample.maf"
    cp "$input" input.maf
    purity=$(/home/sugen/resolveclone-runtime/phylogic/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["purity"])' "${input%.maf}.json")
    printf 'sample_id\tmaf_fn\tseg_fn\tpurity\ttimepoint\n%s\tinput.maf\t\t%s\t0\n' "$sample" "$purity" > input.sif
    if [ "$tool" = phylogic ]; then entry="$root/external/phylogicndt/PhylogicNDT.py"; else entry="$root/scripts/run_phylogic_bracketed.py"; fi
    timeout "$limit" /home/sugen/resolveclone-runtime/phylogic/bin/python "$entry" Cluster -i "$sample" -sif input.sif --maf_input_type calc_ccf --maf --no_html --seed 17 > run.log 2>&1 || code=$?
    ;;
  *) echo "unknown tool $tool" >&2; exit 2 ;;
esac
echo "$code" > exit_code
if [ "$code" -eq 0 ]; then echo ok > status; elif [ "$code" -eq 124 ]; then echo "timeout_${limit}s" > status; else echo error > status; fi
echo "$sample $tool $(cat status)"
