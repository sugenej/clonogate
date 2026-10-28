# Backup tables for the STS report

Optional tables to drop into `STS_REPORT_DRAFT_v2.md` as needed. Every number is copied from `MANUSCRIPT_DRAFT_v4.md` or the result file named under each table; check against that source before use. All results are simulated except Table D.

---

## Table A. Attempts to fix the false-split problem (STS Section 6)

| Attempt | Test | Result | Outcome |
|---|---|---|---|
| Beta-binomial emissions, ρ = 0.01 (gate null recalibrated) | sc2A, dev designs P1–P25, paired vs frozen ClonoGate | −0.032 [−0.060, −0.006] | Worse; under-splits |
| Beta-binomial emissions, ρ = 0.03 | same | −0.115 [−0.164, −0.067] | Worse; 18 of 25 tumors lose a cluster |
| v3.2 at the benchmark's median ρ = 0.0015 | sc2A / sc1B, P1–P25 | −0.010 [−0.020, −0.002] / +0.002 [−0.024, +0.030] | Slightly worse on sc2A |
| Fixed ceiling ρ = 0.02 (v3.2c) | sc2A / sc1B, P1–P25 | −0.084 [−0.126, −0.044] / −0.109 [−0.167, −0.043] | Safe on false splits but the most costly |
| ρ estimated from the data (v3.6, right flank of the top VAF mode) | sc2A / sc1B, P1–P25 | −0.046 [−0.084, −0.013] / −0.062 [−0.118, −0.005] | Worse |
| v3.6 in the stress test | single-clone ρ = 0.02; two-clone δ = 0.25 | 10 of 60 false splits; detection fell from 22 of 60 (ρ = 0) to 14 of 60 (ρ = 0.02) | Partly cut false splits, lost power |
| Multiplicity prior θ = 0.5 (v3.3) | sc2A, P1–P25 | −0.003 [−0.007, +0.001] | Neutral on the benchmark; removes multiplicity false splits in simulation |
| Multiplicity prior θ = 0.25 | same | −0.008 [−0.021, +0.002] | Neutral |
| Continuous bounded M-step + 5 EM restarts | same | −0.001 [−0.007, +0.003] | Neutral; K changed on 4 of 25 |
| Greedy warm-start split/merge selection of K | same | −0.002 [−0.007, +0.003] | Neutral; same K on every design |
| ρ-agnostic shape test (two beta-binomial clusters vs one) | 360 single-clone simulations per ρ | False splits 1/360, 17/360, 36/360 at ρ = 0, 0.005, 0.02 | Holds false splits at 5–10%, but needs x ≈ 1.6 for 50% detection |

Source: `MANUSCRIPT_DRAFT_v4.md` Section 3.6; `reports/v3_falsesplit/SHAPE_TEST.md`. Penalty λ was held at 20 throughout; the beta-binomial variants were not re-tuned.

---

## Table B. Effect of the x ≥ 1 screen by tool (STS Section 6, Figure 6)

Each cell: splits reported / kept by the screen at an assumed ρ = 0 / kept at an assumed ρ = 0.02. Single-clone samples: every split is false. Two-clone samples: every kept split is a real detection.

| Method | Single-clone, ρ = 0 | Single-clone, noisy (ρ > 0) | Two-clone |
|---|---|---|---|
| ClonoGate (v3, frozen) | 21 / 17 / 16 of 135 | 120 / 114 / 66 of 135 | 289 / 282 / 236 of 300 |
| v3.2, matched ρ (oracle) | – | 4 / 2 / 2 of 75 | 115 / 112 / 91 of 150 |
| v3.3, multiplicity prior | 4 / 1 / 1 of 60 | 30 / 23 / 0 of 60 | – |
| CCube | 0 / 0 / 0 of 135 | 58 / 58 / 52 of 135 | 243 / 243 / 207 of 300 |
| PyClone-VI, binomial | 30 / 30 / 29 of 135 | 125 / 125 / 105 of 135 | 299 / 287 / 256 of 300 |
| PyClone-VI, beta-binomial | 1 / 1 / 1 of 135 | 90 / 89 / 88 of 135 | 262 / 248 / 224 of 300 |
| FastClone | 4 / 1 / 1 of 135 | 6 / 5 / 4 of 135 | 69 / 68 / 47 of 300 |
| MyClone | 135 / 92 / 75 of 135 | 134 / 49 / 37 of 134 | 300 / 155 / 75 of 300 |
| PhylogicNDT | 8 / 8 / 8 of 135 | 114 / 114 / 114 of 135 | 295 / 288 / 269 of 300 |

Source: `reports/v3_falsesplit/REPORTING_RULE.md`. The noisy single-clone group pools four simulation families (equal and unequal clones at ρ = 0.02, and the multiplicity and subclonal-copy-number scenarios at ρ = 0.005). MyClone's splits come from its multiplicity convention, not noise.

---

## Table C. Cluster accuracy against the oracle (STS Section 7, Figure 7)

Adjusted Rand index against the true clonal/subclonal split; equal clones, depth {30, 100} × purity {0.4, 0.8} pooled, 15 seeds per cell (60 runs). The oracle uses copy-number-neutral SNVs and knows the true clone positions, the 50/50 mixing and ρ.

| δ, ρ | Oracle | ClonoGate | v3.2 (oracle ρ) | CCube | PyClone-VI binomial | PyClone-VI beta-binomial | PhylogicNDT | FastClone | MyClone |
|---|---|---|---|---|---|---|---|---|---|
| 0.25, ρ 0 | 0.25 | 0.18 | – | 0.15 | 0.20 | 0.13 | 0.19 | 0.07 | 0.04 |
| 0.55, ρ 0 | 0.70 | 0.57 | – | 0.51 | 0.65 | 0.63 | 0.66 | 0.00 | 0.26 |
| 0.25, ρ 0.02 | 0.12 | 0.09 | 0.06 | 0.08 | 0.07 | 0.07 | 0.08 | 0.01 | 0.02 |
| 0.55, ρ 0.02 | 0.48 | 0.35 | 0.40 | 0.28 | 0.33 | 0.36 | 0.33 | 0.03 | 0.15 |

Source: `MANUSCRIPT_DRAFT_v4.md` Section 3.5 and `reports/v3_falsesplit/ACCURACY_RESULTS.md`. Relative to the oracle, the tools counted as usable reach 51–95% in clean data and 55–83% under noise; the oracle's own ARI ranges from 0.05 to 0.97. The comparison is approximate because tools are scored on all SNVs and the oracle on neutral SNVs only.

---

## Table D. Germline read-noise measurement (STS Section 8, Figure 8)

Heterozygous sites (VAF 0.35–0.65, depth ≥ 20 in the selecting sample), SEQC2 HCC1395 tumor/normal pair; beta-binomial fit with free mean; window-bootstrap 95% interval.

| Measured in | Selected in | Sites | Windows | Mean depth | Mean VAF | ρ [95% CI] | Use |
|---|---|---|---|---|---|---|---|
| Tumor | Normal | 14,285 | 85 | 80 | 0.496 | 0.0030 [0.0023, 0.0038] | Main estimate |
| Tumor, split-half balance filter | Normal | 7,088 | 84 | 80 | 0.497 | 0.0026 [0.0019, 0.0034] | Robustness check |
| Normal | Tumor | 41,320 | 508 | 52 | 0.477 | 0.0797 [0.0585, 0.1155] | **Not valid:** tumor-selected sites include clonal somatic SNVs |

| Depth (tumor-measured) | Sites | ρ, all balanced windows | ρ, split-half filter |
|---|---|---|---|
| 40–60 | 1,464 | 0.0055 | 0.0052 |
| 60–80 | 5,970 | 0.0032 | 0.0033 |
| 80+ | 6,770 | 0.0026 | 0.0020 |

Source: `reports/v3_falsesplit/GERMLINE_RHO_multi.md`. One sequencing run; per-depth-bin intervals were not computed.

---

## Table E. Multiplicity ambiguity: false splits by tool (STS Section 4)

Single clone at cellular prevalence 1.0, 90% of SNVs in a 2+1 gain, depth 60, purity 0.7, 700 SNVs, 20 seeds. "m2" is the share of gain SNVs with multiplicity 2 (so m2 = 0 means every extra copy is a single copy). Any K ≥ 2 is a false split.

| Method | m2 0, ρ 0 | m2 0, ρ 0.005 | m2 0.5, ρ 0 | m2 0.5, ρ 0.005 |
|---|---|---|---|---|
| ClonoGate (v3, frozen) | 19/20 | 20/20 | 0/20 | 0/20 |
| v3.3, multiplicity prior (θ 0.5) | 0/20 | 0/20 | 0/20 | 0/20 |
| PyClone-VI, binomial | 16/20 | 20/20 | 0/20 | 6/20 |
| PyClone-VI, binomial, clusters ≥ 2% of SNVs | 3/20 | 20/20 | 0/20 | 0/20 |
| PyClone-VI, beta-binomial | 0/20 | 14/20 | 0/20 | 0/20 |
| PyClone-VI, beta-binomial, clusters ≥ 2% of SNVs | 0/20 | 2/20 | 0/20 | 0/20 |
| CCube | 0/20 | 0/20 | 0/20 | 0/20 |
| FastClone (first 15 seeds) | 0/15 | 1/15 | 0/15 | 2/15 |
| MyClone (first 15 seeds) | 15/15 | 15/15 | 15/15 | 15/15 |
| PhylogicNDT (first 15 seeds) | 0/15 | 9/15 | 0/15 | 0/15 |

Source: `MANUSCRIPT_DRAFT_v4.md` Section 3.3(b); `reports/v3_falsesplit/METHODS_TRACK_RESULTS.md`, section C. The simulator's single-copy gains favour the multiplicity prior, so v3.3's clean row is not evidence that it is safe on the benchmark's designs.

---

## Table F. Clone count (sc1B) and prevalence (sc1C): paired differences (STS Section 3)

Raw official-scorer scores, ClonoGate minus baseline, paired on the designs where each baseline finished. The manuscript reports confidence intervals only where shown; "–" means none was reported.

| Comparator | sc1B, mean difference (all designs) | sc1B, wins / losses / ties | sc1C, mean difference (all designs) | sc1C, wins / losses |
|---|---|---|---|---|
| CCube (n = 40) | +0.082 | 19 / 10 / 11 | −0.003 | 13 / 27 (holdout: 2 / 15, mean −0.007) |
| PhylogicNDT (n = 19) | +0.329 | – | 0.000 | 9 / 10 |
| FastClone (n = 46 on sc1B, 45 on sc1C) | +0.044 [−0.008, +0.096]; holdout +0.099 [+0.026, +0.172] (n = 25) | – | +0.048 | – |
| Object Integration (n = 43 on sc1B, 39 on sc1C) | +0.175 | – | +0.121 | – |

Source: `MANUSCRIPT_DRAFT_v4.md` Section 3.1. ClonoGate is above CCube, PhylogicNDT and Object Integration on sc1B and tied with FastClone over all designs; on sc1C it is not better than CCube or PhylogicNDT.
