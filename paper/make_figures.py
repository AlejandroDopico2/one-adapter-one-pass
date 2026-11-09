"""Figures and main table of the paper. Reads run metrics (runs/) and results/ only.

  figs/paper_ablation.pdf  H1  IN-R 200: final accuracy (actual + staleness up to refit) and forgetting
  figs/paper_pareto.pdf    H3  final accuracy vs inference energy, budget-matched baselines
  figs/paper_vtab.pdf      H2  VTAB-CIL, k domains per task: final accuracy and oracle staleness
  tables/table_main.tex    final / average incremental accuracy / forgetting on IN-R 200, CIFAR-100, VTAB

python paper/make_figures.py
"""

import collections
import glob
import json
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "figs"
TAB = Path(__file__).parent / "tables"
OUT.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

# Ours in a strong blue; modular baselines in warm hues (TUNA, the main rival, in red); controls in grey.
# Blue vs red/orange stays distinct under red-green colour blindness; markers differ as well.
C = {
    "ours": "#1F5FBF",
    "tuna": "#C62D2D",
    "ease": "#E8862A",
    "sema": "#8B62A8",
    "frozen": "#7A7A7A",
}
DARK = {"ours": "#123E80", "grey": "#6E6E6E"}  # edges and hatching
LIGHT = {"ours": "#1F5FBF", "grey": "#C4C4C4"}  # bar fills
TEXT = "#222222"
MK = {"ours": "o", "tuna": "s", "ease": "^", "sema": "D", "frozen": "v"}
WIDTH = 4.8  # in; paper text width = 12.2 cm
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.9,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "hatch.linewidth": 0.6,
        "pdf.fonttype": 42,
    }
)


def collect(pattern, regex):
    """(groups) -> list of metrics dicts, one per seed."""
    out = collections.defaultdict(list)
    for f in glob.glob(str(ROOT / "runs" / pattern / "*" / "metrics.json")):
        m = re.search(regex, f)
        if m:
            out[m.groups()].append(json.loads(Path(f).read_text()))
    return out


def ms(runs, key="final_accuracy"):
    v = [r[key] for r in runs]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


def panel_label(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)


std = collect("std/*/*", r"/std/(\w+?)/(\w+?)_s\d{4}/")
fair = collect("fair/*/*", r"/fair/(\w+?)/(\w+?)_b3_s\d{4}/")

# -- H1: ablation, IN-R 200 ------------------------------------------------------------
refit = json.loads((ROOT / "results/std_refit.json").read_text())
R = collections.defaultdict(lambda: collections.defaultdict(list))
for run, v in refit.items():
    m = re.search(r"inr200_in21k/(\w+?)_s\d{4}/", run)
    if m:
        R[m.group(1)]["actual"].append(v["actual"])
        R[m.group(1)]["refit"].append(v["refit"])
arms = [  # key, label; frozen and no-adapter have fixed features, so refit = actual
    ("none", "none"),
    ("nonecomp", "+comp."),
    ("distill", "+dist."),
    ("distcomp", "+both\n(ours)"),
]
bench = "inr200_in21k"
# Bar panels after (a): metric key, y label, y limits. Plasticity = accuracy on each task right after learning it.
FORGET = ("mean_forgetting", "forgetting (%)", (0, 18))
PLAST = ("plasticity", "plasticity (%)", (78, 90))


def ablation(panels, name):
    n = 1 + len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(WIDTH, 1.75), gridspec_kw={"wspace": 0.32 if n == 2 else 0.42}
    )
    a = axes[0]
    x = np.arange(len(arms))
    for xi, (key, _) in zip(x, arms):
        act, sd = ms(std[(bench, key)])
        ref = st.mean(R[key]["refit"])
        tone = "ours" if key == "distcomp" else "grey"
        face, edge = LIGHT[tone], DARK[tone]
        a.bar(xi, act - 70, bottom=70, width=0.6, color=face, edgecolor=edge, lw=0.5)
        a.errorbar(xi, act, yerr=sd, color=TEXT, lw=0.5, capsize=1.5, capthick=0.5)
        a.bar(
            xi,
            ref - act,
            bottom=act,
            width=0.6,
            facecolor="white",
            edgecolor=edge,
            lw=0.5,
            hatch="////",
        )
        a.text(
            xi,
            ref + 0.15,
            f"{ref - act:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=edge,
        )
        for ax, (metric, _, ylim) in zip(axes[1:], panels):
            v, vsd = ms(std[(bench, key)], metric)
            ax.bar(
                xi,
                v - ylim[0],
                bottom=ylim[0],
                width=0.6,
                color=face,
                edgecolor=edge,
                lw=0.5,
            )
            ax.errorbar(xi, v, yerr=vsd, color=TEXT, lw=0.5, capsize=1.5, capthick=0.5)
    for ax, (metric, ylab, ylim) in zip(
        axes, [("final_accuracy", "final accuracy (%)", (72, 80)), *panels]
    ):
        ref = ms(std[(bench, "frozen")], metric)[0]
        ax.axhline(ref, color=TEXT, lw=0.6, ls=(0, (4, 2)))
        ax.set_xticks(x)
        # Three panels are too narrow for "+both (ours)": one word each, ours is the blue bar.
        short = ["none", "comp.", "dist.", "both"]
        ax.set_xticklabels(
            [lab for _, lab in arms] if n == 2 else short,
            fontsize=6.8 if n == 2 else 6.5,
        )
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.55, len(arms) - 0.45)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor=DARK["grey"],
            lw=0.5,
            hatch="////",
        ),
        plt.Line2D([], [], color=TEXT, lw=0.6, ls=(0, (4, 2))),
    ]
    labels = ["staleness (refit $-$ actual)", "frozen after task 1"]
    if n == 2:
        a.legend(handles, labels, loc="upper right", fontsize=6.5, handlelength=1.2)
    else:  # panels too narrow: one row below
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            fontsize=6.5,
            handlelength=1.6,
            columnspacing=1.5,
        )
    for ax, lab in zip(axes, "abc"):
        panel_label(ax, f"({lab})")
    save(fig, name)


ablation([FORGET], "paper_ablation")
ablation([PLAST, FORGET], "paper_ablation_plast")

# -- H3: accuracy vs inference energy, budget-matched -------------------------------------
# Energy measured alone on one GPU on IN-R 200 (J per 1000 test images). CIFAR uses the same
# 224 px ViT pass, so its energy = measured per-pass energy x passes (T = 20 tasks).
cost = json.loads((ROOT / "results/cost_fair_inr200_in21k.json").read_text())
cost |= json.loads((ROOT / "results/cost_inr200_in21k.json").read_text())
J = {}
for run, c in cost.items():
    m = re.search(r"/(distcomp|tuna_b3|ease_b3|sema_b3)_s1993/", run)
    if m:
        J[m.group(1).removesuffix("_b3")] = (
            c["energy_j_per_1000_img"] / c["full_passes_per_image"]
        )
ib = json.loads((ROOT / "results/inference_budget_fair.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.9), gridspec_kw={"wspace": 0.28})
for ax, bench, title, T, ylim in [
    (axes[0], "inr200_in21k", "ImageNet-R", 10, (73, 79.5)),
    (axes[1], "cifar_in21k", "CIFAR-100", 20, (82.5, 91.5)),
]:
    last = st.mean(r["accuracy"]["last"] for k, r in ib.items() if f"/{bench}/" in k)
    pts = {  # key: (label, kJ / 1000 images, accuracy, label offset in points, ha)
        "ours": (
            "Ours",
            J["distcomp"],
            ms(std[(bench, "distcomp")])[0],
            (5, 0),
            "left",
        ),
        "frozen": (
            "Frozen",
            J["distcomp"],
            ms(std[(bench, "frozen")])[0],
            (-5, 0),
            "right",
        ),
        "sema": ("SEMA", J["sema"], ms(fair[(bench, "sema")])[0], (5, 0), "left"),
        "ease": (
            "EASE",
            J["ease"] * T,
            ms(fair[(bench, "ease")])[0],
            (0, -7),
            "center",
        ),
        "tuna": (
            "TUNA",
            J["tuna"] * (T + 1),
            ms(fair[(bench, "tuna")])[0],
            (0, 6),
            "center",
        ),
    }
    for key, (lab, e, acc, off, ha) in pts.items():
        ax.plot(
            e / 1000,
            acc,
            MK[key],
            color=C[key],
            ms=5 if key == "ours" else 4,
            mec=C[key],
            zorder=4 if key == "ours" else 3,
        )
        ax.annotate(
            lab,
            (e / 1000, acc),
            xytext=off,
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=7.2 if key == "ours" else 6.8,
            color=C["ours"] if key == "ours" else TEXT,
        )
    # Diagnostic: TUNA evaluated with one pass (last snapshot, own head, no refit).
    e1, ef = J["tuna"] / 1000, pts["tuna"][1] / 1000
    ax.plot(
        [e1, ef],
        [last, pts["tuna"][2]],
        color=C["tuna"],
        lw=0.5,
        ls=(0, (2, 2)),
        zorder=1,
    )
    ax.plot(e1, last, MK["tuna"], mfc="white", mec=C["tuna"], ms=4, mew=0.8, zorder=3)
    ax.annotate(
        "TUNA (1 pass)",
        (e1, last),
        xytext=(5, -1),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=6.8,
        color=TEXT,
    )
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10"])
    ax.minorticks_off()
    ax.set_xlim(0.17, 20)
    ax.set_ylim(*ylim)
    ax.set_xlabel("energy (kJ / 1000 test images)")
    ax.set_title(title, fontsize=8)
axes[0].set_ylabel("final accuracy (%)")
save(fig, "paper_pareto")

# -- H2: VTAB, k domains per task (IN21k) ---------------------------------------------------
vt = collect(
    "std/vtab_in21k/*",
    r"/(k\d)_(distcomp|ease|easeb3|tuna|sema|frozen|noadapt|oracle5comp|oracle5)_s\d{4}/",
)
stale = collections.defaultdict(
    list
)  # oracle staleness, from the refit post-processing log
for line in (ROOT / "results/vtab_oracle_staleness.log").read_text().splitlines():
    m = re.match(
        r"runs/std/vtab_in21k/(k\d)_(oracle5comp|oracle5)_s\d{4}/\S+ (\{.*\})", line
    )
    if m:
        stale[m.groups()[:2]].append(json.loads(m.group(3))["staleness"])
series = [  # key, label, colour, marker, linestyle, filled
    ("distcomp", "Ours (shared adapter)", C["ours"], "o", "-", True),
    ("tuna", "TUNA", C["tuna"], "s", "-", True),
    ("ease", "EASE (per task)", C["ease"], "^", "-", True),
    ("oracle5", "Domain-isolated", C["frozen"], "v", "-", True),
    ("oracle5comp", "Domain-isolated + comp.", C["frozen"], "v", (0, (3, 1.5)), False),
]
ks = np.arange(1, 6)
fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 1.8), gridspec_kw={"wspace": 0.3})
for key, lab, col, mk, ls, filled in series:
    m = np.array([ms(vt[(f"k{k}", key)])[0] for k in ks])
    s = np.array([ms(vt[(f"k{k}", key)])[1] for k in ks])
    mfc = col if filled else "white"
    lw = 1.4 if key == "distcomp" else 0.9
    z = 4 if key == "distcomp" else 3
    a.fill_between(ks, m - s, m + s, color=col, alpha=0.15, lw=0, zorder=1)
    a.plot(
        ks, m, color=col, ls=ls, lw=lw, marker=mk, mfc=mfc, mec=col, label=lab, zorder=z
    )
    if key.startswith("oracle"):
        sm = np.array([st.mean(stale[(f"k{k}", key)]) for k in ks])
        ss = np.array([st.stdev(stale[(f"k{k}", key)]) for k in ks])
        b.fill_between(ks, sm - ss, sm + ss, color=col, alpha=0.15, lw=0, zorder=1)
        b.plot(ks, sm, color=col, ls=ls, marker=mk, mfc=mfc, mec=col, zorder=3)
for ax in (a, b):
    ax.set_xticks(ks)
    ax.set_xlim(0.7, 5.3)
    ax.set_xlabel("domains per task $k$")
a.set_ylim(74, 96)
a.set_ylabel("final accuracy (%)")
b.set_ylim(-0.5, 21)
b.set_ylabel("staleness (%)")
handles, labels = a.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.07),
    ncol=3,
    fontsize=6.4,
    handlelength=1.8,
    columnspacing=0.9,
    handletextpad=0.35,
)
panel_label(a, "(a)")
panel_label(b, "(b)")
save(fig, "paper_vtab")

# -- Table 1: standard metrics, budget-matched baselines -----------------------------------
rows = [  # label, passes, {bench: runs}
    (
        "No adapter",
        "1",
        {b: std[(b, "noadapt")] for b in ("inr200_in21k", "cifar_in21k")}
        | {"vtab": vt[("k1", "noadapt")]},
    ),
    (
        "Frozen after $t_1$",
        "1",
        {b: std[(b, "frozen")] for b in ("inr200_in21k", "cifar_in21k")}
        | {"vtab": vt[("k1", "frozen")]},
    ),
    (
        "SEMA",
        "1",
        {b: fair[(b, "sema")] for b in ("inr200_in21k", "cifar_in21k")}
        | {"vtab": vt[("k1", "sema")]},
    ),
    (
        "EASE",
        "$T$",
        {b: fair[(b, "ease")] for b in ("inr200_in21k", "cifar_in21k")}
        | {"vtab": vt[("k1", "easeb3")]},
    ),
    (
        "TUNA",
        "$T{+}1$",
        {b: fair[(b, "tuna")] for b in ("inr200_in21k", "cifar_in21k")}
        | {"vtab": vt[("k1", "tuna")]},
    ),
    (
        "Ours",
        "1",
        {b: std[(b, "distcomp")] for b in ("inr200_in21k", "cifar_in21k")}
        | {"vtab": vt[("k1", "distcomp")]},
    ),
]
cols = ("inr200_in21k", "cifar_in21k", "vtab")
best = {}
for bch in cols:
    for key, sign in (
        ("final_accuracy", 1),
        ("mean_incremental_accuracy", 1),
        ("mean_forgetting", -1),
    ):
        vals = [sign * ms(r[2][bch], key)[0] for r in rows if r[2].get(bch)]
        best[(bch, key)] = sign * max(vals)
lines = [
    r"\begin{table}[t]",
    r"\centering\footnotesize",
    r"\setlength{\tabcolsep}{3.2pt}",
    r"\begin{tabular}{lc ccc ccc ccc}",
    r"\toprule",
    r" & & \multicolumn{3}{c}{ImageNet-R (10 tasks)} & \multicolumn{3}{c}{CIFAR-100 (20 tasks)} & \multicolumn{3}{c}{VTAB (5 tasks)} \\",
    r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
    r"Method & Passes & $A_T\uparrow$ & $\bar A\uparrow$ & $F\downarrow$ & $A_T\uparrow$ & $\bar A\uparrow$ & $F\downarrow$ & $A_T\uparrow$ & $\bar A\uparrow$ & $F\downarrow$ \\",
    r"\midrule",
]
for i, (lab, passes, runs) in enumerate(rows):
    if lab == "Ours":
        lines.append(r"\midrule")
    cells = [lab, passes]
    for bch in cols:
        r = runs.get(bch)
        if not r:
            cells += ["--"] * 3
            continue
        for key in ("final_accuracy", "mean_incremental_accuracy", "mean_forgetting"):
            m, s = ms(r, key)
            c = f"{m:.1f}"
            if key == "final_accuracy":
                c += rf"{{\scriptsize$\pm${s:.1f}}}"
            if abs(m - best[(bch, key)]) < 1e-9:
                c = rf"\textbf{{{c}}}"
            cells.append(c)
    lines.append(" & ".join(cells) + r" \\")
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Final accuracy $A_T$, average incremental accuracy $\bar A$ and forgetting $F$ (\%, mean over 3 class orders). "
    r"ViT-B/16 (IN21k); all adapters on blocks 3--11 with bottleneck 16. Passes: ViT forward passes per test image. "
    r"VTAB: one domain per task.}",
    r"\label{tab:main}",
    r"\end{table}",
]
(TAB / "table_main.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("written to", OUT, TAB)
