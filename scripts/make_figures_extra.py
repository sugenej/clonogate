"""Extra STS figures: ClonoGate pipeline, false splits by tool, noise/subclone look-alike histograms.

False-split counts come from the Section 4 table of reports/STS_REPORT_DRAFT.md (technical numbers in MANUSCRIPT_DRAFT_v4.md).
The look-alike figure is an illustrative simulation (fixed seed), not a result: a single clone with beta-binomial overdispersion
rho_equiv = f(1-f) dq^2 / (q(1-q)) versus a two-clone tumor with binomial reads. Writes reports/figures/fig_*.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "reports" / "figures"
BLUE, RED, GREY = "#1b6ca8", "#b3261e", "#6b7280"

# Section 4 table: (false splits, runs) at rho = 0 and rho = 0.02
FALSE_SPLITS = {
    "ClonoGate": ((0, 60), (60, 60)),
    "CCube": ((0, 60), (39, 60)),
    "PyClone-VI\nbinomial": ((4, 60), (60, 60)),
    "PyClone-VI\nbeta-binomial": ((0, 60), (58, 60)),
    "PhylogicNDT": ((1, 60), (60, 60)),
    "FastClone": ((2, 60), (2, 60)),
    "MyClone*": ((60, 60), (59, 59)),
}


def fig_false_splits() -> None:
    names = list(FALSE_SPLITS)
    clean = [100 * k / n for (k, n), _ in FALSE_SPLITS.values()]
    noisy = [100 * k / n for _, (k, n) in FALSE_SPLITS.values()]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(x - 0.2, clean, 0.4, color=BLUE, label="rho = 0 (binomial reads)")
    ax.bar(x + 0.2, noisy, 0.4, color=RED, label="rho = 0.02 (overdispersed reads)")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("single-clone tumors falsely split (%)")
    ax.set_ylim(0, 122)
    ax.legend(loc="upper center", ncol=2, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "*MyClone splits every tumor because of its multiplicity convention; see text.", fontsize=7, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_false_splits.png", dpi=200)
    plt.close(fig)


def fig_lookalike() -> None:
    purity, depth, n, delta, frac = 0.6, 60, 700, 0.25, 0.5
    q = purity / 2
    dq = delta * purity / 2
    rho_eq = frac * (1 - frac) * dq**2 / (q * (1 - q))
    rng = np.random.default_rng(20261024)
    d = np.maximum(rng.poisson(depth, n), 8)
    a_q = q * (1 / rho_eq - 1)
    single = rng.binomial(d, rng.beta(a_q, (1 - q) * (1 / rho_eq - 1), n)) / d
    sub = rng.random(n) < frac
    two = rng.binomial(d, np.where(sub, q - dq, q)) / d
    bins = np.linspace(0.08, 0.5, 45)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    panels = [(single, f"One clone, overdispersed reads\n(rho = {rho_eq:.4f})", RED),
              (two, f"Two clones, binomial reads\n(subclone at {int(frac * 100)}% of SNVs, gap {dq:.3f})", BLUE)]
    for ax, (v, title, color) in zip(axes, panels):
        ax.hist(v, bins=bins, color=color, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("variant allele fraction")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.97, 0.93, f"SD = {v.std():.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=GREY)
    axes[0].set_ylabel("number of SNVs")
    fig.tight_layout()
    fig.savefig(OUT / "fig_lookalike.png", dpi=200)
    plt.close(fig)
    print(f"rho_equiv={rho_eq:.5f}; SD single={single.std():.4f}, two={two.std():.4f}")


def _box(ax, xy, w, h, text, face, fs=8.5):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=face, ec=GREY, lw=1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    light, mid, warn = "#eaf2f8", "#fdf2e0", "#fbe9e7"
    w, h, y = 2.1, 1.5, 1.7
    _box(ax, (0.1, y), w, h, "1. Input per SNV\nalt reads a, depth n,\ncopy number,\npurity p", light, fs=8)
    _box(ax, (2.6, y), w, h, "2. EM binomial mixture\nK = 1..6, VAF = mu c_k / D\n(multiplicity summed out)", light, fs=8)
    _box(ax, (5.1, y), w, h, "3. Penalised choice of K\nloglik - 20 (2K - 1)", light, fs=8)
    _box(ax, (7.6, y), w, h, "4. Bootstrap gate\non CN-neutral SNVs\nT vs 300 binomial nulls\np > 0.02: K = 1\np <= 0.02: K >= 2 allowed", mid, fs=7.5)
    _box(ax, (10.1, y), 1.8, h, "5. Refit and\nassign every SNV\nto a clone", light, fs=8)
    for x0 in (2.2, 4.7, 7.2, 9.7):
        _arrow(ax, (x0, y + h / 2), (x0 + 0.4, y + h / 2))
    _box(ax, (4.6, 0.1), 5.2, 0.9, "Optional post-hoc screen, usable on any tool's output:\nkeep a subclone only if its standardised gap x >= 1", warn, fs=8)
    ax.text(6.0, 3.85, "ClonoGate: a significance gate on top of a copy-number-aware mixture", fontsize=11, ha="center", weight="bold")
    fig.savefig(OUT / "fig_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# reports/v3_falsesplit/REPORTING_RULE.md, assumed rho .02: (false splits reported, kept) on noisy single-clone tumors and
# (detections reported, kept) on two-clone tumors. MyClone is omitted (its splits come from its multiplicity convention).
SCREEN = {
    "ClonoGate": ((120, 66), (289, 236)),
    "CCube": ((58, 52), (243, 207)),
    "PyClone-VI\nbinomial": ((125, 105), (299, 256)),
    "PyClone-VI\nbeta-binomial": ((90, 88), (262, 224)),
    "PhylogicNDT": ((114, 114), (295, 269)),
    "FastClone": ((6, 4), (69, 47)),
}


def fig_screen() -> None:
    names = list(SCREEN)
    removed = [100 * (f[0] - f[1]) / f[0] for f, _ in SCREEN.values()]
    lost = [100 * (t[0] - t[1]) / t[0] for _, t in SCREEN.values()]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(x - 0.2, removed, 0.4, color=BLUE, label="false splits removed")
    ax.bar(x + 0.2, lost, 0.4, color=RED, label="true detections lost")
    for xi, v in zip(x - 0.2, removed):
        ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    for xi, v in zip(x + 0.2, lost):
        ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("% of the tool's reported splits")
    ax.set_ylim(0, 60)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "Screen: keep a subclone only if x >= 1, assuming rho = 0.02. MyClone omitted (multiplicity artifact).",
             fontsize=7, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_screen_effect.png", dpi=200)
    plt.close(fig)


# reports/v3_falsesplit/GERMLINE_RHO_multi.md, measured in tumour (sites selected in the normal)
GERM_DEPTH = ["40-60", "60-80", "80+"]
GERM_RHO = [0.0055, 0.0032, 0.0026]
GERM_RHO_SPLIT = [0.0052, 0.0033, 0.0020]
GERM_SITES = [1464, 5970, 6770]


def fig_germline() -> None:
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(x - 0.2, GERM_RHO, 0.4, color=BLUE, label="all balanced windows")
    ax.bar(x + 0.2, GERM_RHO_SPLIT, 0.4, color="#7aa6c9", label="split-half filter")
    ax.axhspan(0.0023, 0.0038, color=GREY, alpha=0.18, label="overall rho 0.0030 [0.0023, 0.0038]")
    ax.axhline(0.005, color="#e08a1e", ls="--", lw=1.2, label="simulated rho = 0.005")
    ax.set_xticks(x, [f"{d}\n({n:,} sites)" for d, n in zip(GERM_DEPTH, GERM_SITES)], fontsize=8)
    ax.set_xlabel("read depth at the site")
    ax.set_ylabel("overdispersion rho (beta-binomial fit)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "SEQC2 HCC1395 tumor, germline heterozygous sites (true VAF 0.5). Per-bin intervals not computed.",
             fontsize=7, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_germline_rho_depth.png", dpi=200)
    plt.close(fig)


# STS draft Section 3 table: sc2A paired difference, ClonoGate minus baseline: (estimate, low, high, n paired)
FOREST = {
    "CCube": {"Holdout": (0.034, -0.006, 0.088, 17), "All designs": (0.031, 0.007, 0.062, 40)},
    "PhylogicNDT": {"Holdout": (0.028, 0.001, 0.053, 7), "All designs": (0.038, 0.023, 0.052, 19)},
    "FastClone": {"Holdout": (0.114, 0.073, 0.158, 25), "All designs": (0.092, 0.063, 0.121, 46)},
    "Object Integration": {"Holdout": (0.242, 0.181, 0.303, 23), "All designs": (0.323, 0.269, 0.376, 43)},
}


def fig_forest() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    y, ticks, labels = 0.0, [], []
    for name, rows in reversed(list(FOREST.items())):
        for subset, color, dy in (("All designs", BLUE, -0.17), ("Holdout", RED, 0.17)):
            est, lo, hi, n = rows[subset]
            ax.errorbar(est, y + dy, xerr=[[est - lo], [hi - est]], fmt="o", color=color, capsize=3, ms=5,
                        label=subset if name == "CCube" else None)
            ax.text(0.385, y + dy, f"n = {n}", va="center", fontsize=7, color=GREY)
        ticks.append(y)
        labels.append(name)
        y += 1
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ticks, labels)
    ax.set_xlim(-0.05, 0.43)
    ax.set_xlabel("sc2A paired difference, ClonoGate minus baseline (95% CI)")
    ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(0.97, 0.52))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_forest_sc2A.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_false_splits()
    fig_lookalike()
    fig_pipeline()
    fig_screen()
    fig_germline()
    fig_forest()
