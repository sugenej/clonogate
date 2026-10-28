"""Manuscript figures: (1) detection probability vs standardised gap, (2) method ARI vs oracle ceiling.

Fig 1 reads the cells used by scripts/fit_boundary.py; the curve is the 192-cell fit in BOUNDARY_FIT.md.
Fig 2 uses the section 3.5 table of MANUSCRIPT_DRAFT_v4.md (method ARI from reports/v3_falsesplit/ACCURACY_RESULTS.md,
oracle ARI from scripts/oracle_ari.py; the oracle uses CN-neutral SNVs only, so the comparison is approximate).
Writes reports/figures/fig1_detection_vs_x.png and fig2_ari_vs_oracle.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))
from fit_boundary import load_cells  # noqa: E402

OUT = ROOT / "reports" / "figures"
# 192-cell fit (BOUNDARY_FIT.md): logit P = B0 + B1*log(x_eff), x_eff = x (N/700)^A (fs/0.5)^C
B0, B1, A, C = 2.36, 5.95, 0.24, 0.38
RHO_COLORS = {0.0: "#1b6ca8", 0.005: "#e08a1e", 0.02: "#b3261e"}

# Section 3.5 table: (delta, rho) -> oracle ARI and per-method ARI
ARI_CELLS = ["delta .25\nrho 0", "delta .55\nrho 0", "delta .25\nrho .02", "delta .55\nrho .02"]
ORACLE = [0.25, 0.70, 0.12, 0.48]
METHOD_ARI = {
    "v3": [0.18, 0.57, 0.09, 0.35],
    "CCube": [0.15, 0.51, 0.08, 0.28],
    "PyClone-VI binomial": [0.20, 0.65, 0.07, 0.33],
    "PyClone-VI beta-binomial": [0.13, 0.63, 0.07, 0.36],
    "PhylogicNDT": [0.19, 0.66, 0.08, 0.33],
    "FastClone": [0.07, 0.00, 0.01, 0.03],
    "MyClone": [0.04, 0.26, 0.02, 0.15],
}


def fig_detection() -> None:
    # the 192 cells of the fit: beta-binomial generator, single-copy gains only
    cells = [c for c in load_cells() if c["generator"] == "bb" and c["mult2"] == 0.0]
    assert len(cells) == sum(abs(c["rho"] - r) < 1e-9 for c in cells for r in RHO_COLORS), "cell with an unplotted rho"
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for rho, color in RHO_COLORS.items():
        sel = [c for c in cells if abs(c["rho"] - rho) < 1e-9]
        if not sel:
            continue
        x = [c["x"] * (c["n_snv"] / 700) ** A * (c["frac_sub"] / 0.5) ** C for c in sel]
        p = [c["hits"] / c["n"] for c in sel]
        ax.scatter(x, p, s=[c["n"] for c in sel], color=color, alpha=0.55, edgecolor="none", label=f"rho = {rho:g}")
    grid = np.geomspace(0.1, 3.5, 200)
    ax.plot(grid, 1 / (1 + np.exp(-(B0 + B1 * np.log(grid)))), color="black", lw=1.6, label="logistic fit")
    ax.axvline(1.0, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("standardised gap, effective x")
    ax.set_ylabel("fraction of simulations with K >= 2")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_detection_vs_x.png", dpi=200)
    plt.close(fig)


def fig_ari() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    names = list(METHOD_ARI)
    width = 0.8 / (len(names) + 1)
    pos = np.arange(len(ARI_CELLS))
    ax.bar(pos - 0.4 + width / 2, ORACLE, width, color="black", label="oracle")
    for i, name in enumerate(names, start=1):
        ax.bar(pos - 0.4 + width / 2 + i * width, METHOD_ARI[name], width, label=name)
    ax.set_xticks(pos)
    ax.set_xticklabels(ARI_CELLS)
    ax.set_ylabel("ARI vs true clonal/subclonal split")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_ari_vs_oracle.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_detection()
    fig_ari()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
