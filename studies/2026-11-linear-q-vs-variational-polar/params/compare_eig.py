"""Figures and tables of the eigenvalue blocks (eig_block.py, run_eig_all.sh), read beside this file (or from $STUDY_RUNS):

- eig_stable_spectrum.png: spectrum of the stable screw pinch, block (1, -1), dt = 0.125: |omega| against the radial
  peak of the eigenvector for both models, with the shear Alfven and slow continua; growing eigenvalues marked.
- eig_spurious.png: growth rate of the growing eigenvalues against |omega| dt, stable case, all dt, both models.
- eig_dt_convergence.png: growth rate of the physical instability (kink, Suydam) against dt for both models, and its
  relative error against the dt -> 0 limit (Richardson, second order).
- eig_tables.md: all numbers of the figures, the axis mode (var_pi) and the radial convergence (r32).
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("STUDY_RUNS", HERE))
C = {"q": "#2a78d6", "var": "#eb6834", "var_pi": "#1baf7a"}
LABEL = {"q": "q model (LinearMHDWithQ)", "var": "variational, L2 transport", "var_pi": "variational, commuting"}
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#52514e",
                     "axes.grid": True, "grid.color": "#e4e3df", "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 11})


def load(model, case, dt, suffix=""):
    f = RUNS / f"eig_{model}_{case}_dt{dt}{suffix}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    for k in ("gamma", "omega", "rpeak"):
        d[k] = np.asarray(d[k])
    return d


def physical(d, tol=1e-6):
    """The largest real growing eigenvalue (omega = 0): the physical instability of the block."""
    real = (np.abs(d["omega"]) < 1e-6) & (d["gamma"] > tol)
    return float(d["gamma"][real].max()) if real.any() else 0.0


def spurious(d, tol=1e-6):
    """The growing eigenvalues with omega != 0."""
    s = (np.abs(d["omega"]) >= 1e-6) & (d["gamma"] > tol)
    return d["gamma"][s], np.abs(d["omega"][s]) * d["dt"]


lines = []

# ---------------------------------------------------------------- stable spectrum with the continua
from cases import CASES  # noqa: E402

c = CASES["stable"]
r = np.linspace(0.0, 1.0, 400)
q = c["q0"] + (c["q1"] - c["q0"]) * r**2
B2 = 1.0 + (r / (q * c["R0"])) ** 2
p = c["p0"] + c["q0"] / (2 * c["R0"] ** 2 * (c["q1"] - c["q0"])) * (1 / q**2 - 1 / c["q1"] ** 2)
wA = np.abs(1.0 / q - 1.0) / c["R0"]
wS = wA * np.sqrt(5 / 3 * p / (5 / 3 * p + B2))
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, m in zip(axs, ("q", "var")):
    d = load(m, "stable", 0.125)
    if d is None:
        continue
    w = np.abs(d["omega"])
    osc = w > 1e-8  # the static modes (omega = 0) are left out
    grow = (d["gamma"] > 1e-6) & osc
    ax.plot(r, wA, color="#0b0b0b", lw=1.6, label="shear Alfven continuum")
    ax.plot(r, wS, color="#52514e", lw=1.4, ls=(0, (4, 3)), label="slow continuum")
    ax.scatter(d["rpeak"][osc & ~grow], w[osc & ~grow], s=14, color=C[m], label="eigenvalues (not growing)")
    if grow.any():
        ax.scatter(d["rpeak"][grow], w[grow], s=40, facecolor="none", edgecolor="#d6312a", lw=1.6,
                   label="growing (spurious)")
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 40)
    ax.text(0.02, 0.97, f"{int((~osc).sum())} static modes (omega = 0) not shown", transform=ax.transAxes, fontsize=8,
            va="top", color="#52514e")
    ax.set_xlabel("radial peak of the eigenvector")
    ax.set_title(LABEL[m] + ", dt = 0.125", loc="left", fontsize=10)
axs[0].set_ylabel("|omega|")
axs[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Stable screw pinch (q = 1.1 -> 1.9), block (m, n) = (1, -1), polar cylinder 16 x 12 x 12", x=0.01,
             ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(RUNS / "eig_stable_spectrum.png", dpi=150)

# ---------------------------------------------------------------- spurious growth against |omega| dt (stable case)
fig, ax = plt.subplots(figsize=(8, 4.2))
mk = {0.5: "o", 0.25: "s", 0.125: "^"}
lines.append("### Stable case: growing eigenvalues (all spurious)\n\n| model | dt | number | max gamma | |omega| dt of the max |\n|---|---|---|---|---|")
for m in ("q", "var"):
    for dt in (0.5, 0.25, 0.125):
        d = load(m, "stable", dt)
        if d is None:
            continue
        g, wdt = spurious(d)
        if g.size:
            ax.scatter(wdt, g, marker=mk[dt], color=C[m], s=30, label=f"{LABEL[m]}, dt = {dt}")
        lines.append(f"| {m} | {dt} | {g.size // 2} pairs | {g.max() if g.size else d['gamma'].max():.1e} | "
                     f"{wdt[np.argmax(g)] if g.size else float('nan'):.2f} |")
ax.set_yscale("log")
ax.set_xlabel("|omega| dt")
ax.set_ylabel("growth rate gamma")
ax.set_title("Stable screw pinch: growing eigenvalues (the variational scheme has none)", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(RUNS / "eig_spurious.png", dpi=150)

# ---------------------------------------------------------------- dt convergence of the physical instabilities
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for case, ax in zip(("kink", "suydam"), axs):
    lines.append(f"\n### {case}: growth rate of the instability and largest spurious growth\n\n"
                 "| model | dt | gamma | error vs dt -> 0 | max spurious gamma | leak |\n|---|---|---|---|---|---|")
    for m in ("q", "var"):
        dts, gs, sp, lk = [], [], [], []
        for dt in (1.0, 0.5, 0.25, 0.125, 0.0625):
            d = load(m, case, dt)
            if d is None:
                continue
            dts.append(dt)
            gs.append(physical(d))
            g, _ = spurious(d)
            sp.append(g.max() if g.size else 0.0)
            lk.append(d["leak"])
        if len(gs) < 2:
            continue
        dts, gs = np.array(dts), np.array(gs)
        g_lim = gs[-1] + (gs[-1] - gs[-2]) / ((dts[-2] / dts[-1]) ** 2 - 1)  # Richardson, second order
        err = np.abs(gs - g_lim) / g_lim
        ax.loglog(dts, err, "o-", color=C[m], lw=2, label=f"{LABEL[m]} (limit {g_lim:.4f})")
        for dt, g, e, s, l in zip(dts, gs, err, sp, lk):
            lines.append(f"| {m} | {dt} | {g:.5f} | {e:.1e} | {s:.1e} | {l:.0e} |")
    ax.loglog([0.125, 1.0], [2e-3, 2e-3 * 64], color="#52514e", ls=":", lw=1, label="slope 2")
    ax.set_xlabel("dt")
    ax.set_ylabel("relative error of gamma")
    ax.set_title({"kink": "Kink (1, -1), q = 0.7 -> 1.3", "suydam": "Suydam (2, -1), q = 1.5 -> 2.5"}[case],
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(RUNS / "eig_dt_convergence.png", dpi=150)

# ---------------------------------------------------------------- axis mode and radial convergence
lines.append("\n### Kink, dt = 0.25: transport projection and radial resolution\n\n| run | largest growth rate | "
             "physical kink gamma | radial peak of the fastest mode |\n|---|---|---|---|")
for m, suffix, name in (("var_pi", "", "variational, commuting, 16 radial"), ("var", "", "variational, L2, 16 radial"),
                        ("var", "_r32", "variational, L2, 32 radial"), ("q", "", "q model, 16 radial"),
                        ("q", "_r32", "q model, 32 radial")):
    d = load(m, "kink", 0.25, suffix)
    if d is None:
        continue
    i = int(np.argmax(d["gamma"]))
    lines.append(f"| {name} | {d['gamma'][i]:.5f} | {physical(d):.5f} | {d['rpeak'][i]:.2f} |")
lines.append("\n### Suydam, dt = 0.125: radial resolution\n\n| model | 16 radial | 32 radial |\n|---|---|---|")
for m in ("q", "var"):
    a16, a32 = load(m, "suydam", 0.125), load(m, "suydam", 0.125, "_r32")
    if a16 is not None and a32 is not None:
        lines.append(f"| {m} | {physical(a16):.5f} | {physical(a32):.5f} |")
# ---------------------------------------------------------------- polar torus (AdhocTorus, not an equilibrium)
lines.append("\n### Polar torus (AdhocTorus, not in force balance), block n = -1\n\n| model | dt | real growing modes | "
             "growing with |omega| dt > 1: number, max gamma | other growing pairs: max gamma | leak |\n"
             "|---|---|---|---|---|---|")
fig, ax = plt.subplots(figsize=(8, 4.2))
for m in ("q", "var"):
    pts = []
    for dt in (0.5, 0.25, 0.125):
        d = load(m, "torus", dt)
        if d is None:
            continue
        g, w = d["gamma"], np.abs(d["omega"])
        real = np.sort(g[(w < 1e-6) & (g > 1e-5)])[::-1][:3]
        hf = (g > 1e-5) & (w * dt > 1)
        lf = (g > 1e-5) & (w >= 1e-6) & (w * dt <= 1)
        lines.append(f"| {m} | {dt} | {', '.join(f'{x:.5f}' for x in real)} | {hf.sum() // 2}, "
                     f"{g[hf].max() if hf.any() else 0:.1e} | {g[lf].max() if lf.any() else 0:.1e} | {d['leak']:.0e} |")
        pts.append((dt, g[hf].max() if hf.any() else 1e-14, g[lf].max() if lf.any() else 1e-14))
    if pts:
        pts = np.array(pts)
        ax.loglog(pts[:, 0], pts[:, 1], "o-", color=C[m], lw=2, label=f"{LABEL[m]}: growing, |omega| dt > 1")
        ax.loglog(pts[:, 0], pts[:, 2], "s--", color=C[m], lw=1.5, label=f"{LABEL[m]}: other growing pairs")
ax.set_xlabel("dt")
ax.set_ylabel("largest spurious growth rate")
ax.set_ylim(1e-6, 2e-2)
ax.set_title("Polar torus (AdhocTorus), n = -1: spurious growth (kink: gamma ~ 0.016)", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(RUNS / "eig_torus_spurious.png", dpi=150)

open(RUNS / "eig_tables.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
