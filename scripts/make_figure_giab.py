"""Figure for Section 8.2: the real-read (GIAB HG002 + HG003 computational mixture) check.

Left: observed split rate in the two-cluster cells against the standardised gap x, for ClonoGate v3 and for v3.2 given the measured rho,
with the frozen detection law.  Middle: false-split rate in the single-class controls by depth.  Right: share of v3.2 gate p-values
below alpha = 0.02 on real reads and on reads simulated from fitted noise models (reports/giab_mixture_check/HEAVY_TAIL_CHECK.md).
Writes reports/figures/fig_giab_real_reads.png.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))
from hapmap_check import wilson  # noqa: E402

DATA = ROOT / "reports" / "giab_mixture_check"
OUT = ROOT / "reports" / "figures" / "fig_giab_real_reads.png"
BLUE, RED, GREY = "#1b6ca8", "#b3261e", "#6b7280"
B0, B1 = 2.36, 5.95                                       # frozen law


def load(name: str) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()])


def pair_cells(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (f, dt), g in runs[runs.kind.str.startswith("pair")].groupby(["f", "depth_target"]):
        q1, q2 = np.mean([c[0]["vaf"] for c in g.classes]), np.mean([c[1]["vaf"] for c in g.classes])
        rho = np.mean([(c[0]["rho"] + c[1]["rho"]) / 2 for c in g.classes])
        x = (q1 - q2) / np.sqrt(q1 * (1 - q1) * (1 / g.mean_depth.mean() + rho))
        lo, hi = wilson(int(g.split.sum()), len(g))
        rows.append(dict(x=x, rate=g.split.mean(), lo=lo, hi=hi))
    return pd.DataFrame(rows).sort_values("x")


def control_rates(runs: pd.DataFrame) -> pd.DataFrame:
    c = runs[runs.kind == "control"]
    rows = []
    for dt, g in c.groupby("depth_target"):
        lo, hi = wilson(int(g.split.sum()), len(g))
        rows.append(dict(depth=dt, rate=g.split.mean(), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def gate_shares() -> list:
    text = (DATA / "HEAVY_TAIL_CHECK.md").read_text()
    out = []
    for label, key in (("real HG002\nreads", "real HG002 reads"), ("simulated:\nbeta-binomial", "single beta-binomial"),
                       ("simulated:\n2-component\nmixture", "two-component mixture")):
        m = re.search(re.escape(key) + r"[^|]*\| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", text)
        out.append((label, float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return out


def main() -> None:
    v3, v32 = load("runs.jsonl"), load("runs_v32.jsonl")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), gridspec_kw=dict(width_ratios=[1.35, 1, 0.9]))

    ax = axes[0]
    xs = np.linspace(0.2, 1.6, 200)
    ax.plot(xs, 1 / (1 + np.exp(-(B0 + B1 * np.log(xs)))), color="black", lw=1.6, label="frozen law")
    for runs, colour, label, dx in ((v3, BLUE, "v3 (binomial gate)", -0.008), (v32, RED, "v3.2, measured ρ supplied", 0.008)):
        c = pair_cells(runs)
        ax.errorbar(c.x + dx, c.rate, yerr=[c.rate - c.lo, c.hi - c.rate], fmt="o", ms=4.5, color=colour, capsize=2, lw=1, label=label)
    ax.set(xlabel="standardised gap x (observed, per cell)", ylabel="fraction of runs with K ≥ 2", ylim=(-0.03, 1.05), title="Two-cluster mixtures")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax = axes[1]
    for runs, colour, label, dx in ((v3, BLUE, "v3", -0.9), (v32, RED, "v3.2, measured ρ", 0.9)):
        c = control_rates(runs)
        ax.errorbar(c.depth + dx, c.rate, yerr=[c.rate - c.lo, c.hi - c.rate], fmt="o-", ms=4.5, color=colour, capsize=2, lw=1, label=label)
    ax.axhline(0.02, color=GREY, ls="--", lw=1)
    ax.text(47, 0.045, "nominal 2%", color=GREY, fontsize=8)
    ax.set(xlabel="mean depth", ylabel="false-split rate (single class)", ylim=(-0.03, 1.05), xticks=[20, 30, 45, 60], title="Single-class controls")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax = axes[2]
    g = gate_shares()
    x = np.arange(len(g))
    ax.bar(x, [v for _, v, _, _ in g], color=[RED, GREY, GREY], width=0.6)
    ax.errorbar(x, [v for _, v, _, _ in g], yerr=[[v - lo for _, v, lo, _ in g], [hi - v for _, v, _, hi in g]], fmt="none", color="black", capsize=3, lw=1)
    ax.axhline(0.02, color=GREY, ls="--", lw=1)
    ax.set(xticks=x, xticklabels=[l for l, *_ in g], ylabel="gate p < 0.02 (depth 60)", ylim=(0, 1.0), title="Gate calibration (v3.2)")
    ax.tick_params(axis="x", labelsize=8)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
