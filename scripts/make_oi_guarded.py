"""Write a guarded copy of the Object Integration submission (OIcopyBattMutect.r); the vendor file is never modified.

The submitted code crashes on 8 of 51 benchmark designs with no fallback for degenerate input.  Its own convention for "cannot
estimate" is to return NA and fall through to the next cellularity estimate, so the guards route the crashes into that path and
change nothing when the original code runs:
  1. batt.cell.lrr: if every clonal segment is copy-neutral (cx all 0) the update is 0/0 -> return NA (the function already returns NA
     when no segment qualifies).                                                              [P4, P19, T15]
  2. batt.plcl: the regression on LogR - mu is attempted while mu is still NA (its first test failed) and lm() has 0 valid rows ->
     require !is.na(mu) before that regression.                                                [P7, P11, P6, T1, T8]
  3. sort(rhot) silently drops NA, so rhos[3] - rhos[1] becomes NA and `if` fails -> sort with NA last and test with isTRUE().
Usage: py -3 scripts/make_oi_guarded.py   (writes scripts/oi_guarded/OIcopyBattMutect_guarded.r)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'external' / 'submission-6184761' / 'OIcopyBattMutect' / 'OIcopyBattMutect.r'
TARGET = ROOT / 'scripts' / 'oi_guarded' / 'OIcopyBattMutect_guarded.r'

# (original text, replacement, expected occurrences)
EDITS = [
    ('logr <- b$LogR[ix]; cx <- (b$nMaj1_A[ix]+b$nMin1_A[ix])/2-1; w <- b$endpos[ix]-b$startpos[ix]+1; w<-w/sum(w); mu<- .0; rho<-.5;\n\tfor(j in 1:1000) {\n\t\tmu.n <- sum((logr-log2(1+cx*rho))*w);',
     'logr <- b$LogR[ix]; cx <- (b$nMaj1_A[ix]+b$nMin1_A[ix])/2-1; w <- b$endpos[ix]-b$startpos[ix]+1; w<-w/sum(w); mu<- .0; rho<-.5;\n\tif(all(cx==0)) return(NA);\n\tfor(j in 1:1000) {\n\t\tmu.n <- sum((logr-log2(1+cx*rho))*w);', 1),
    ('if(length(unique(b$cnA[ix]))>1) {', 'if(!is.na(mu) && length(unique(b$cnA[ix]))>1) {', 1),
    ('rhos <- sort(rhot);', 'rhos <- sort(rhot, na.last=TRUE);', 3),
    ('if(rhos[3]-rhos[1]<.1)', 'if(isTRUE(rhos[3]-rhos[1]<.1))', 3),
]


def main() -> None:
    text = SOURCE.read_text()
    for old, new, count in EDITS:
        found = text.count(old)
        assert found == count, f'expected {count} occurrence(s) of {old[:50]!r}, found {found}'
        text = text.replace(old, new)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text)
    print('wrote', TARGET)


if __name__ == '__main__':
    main()
