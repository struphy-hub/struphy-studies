"""Kink time runs at a large time step (params_linear.py --case kink --dt 0.5 --save-velocity): does the time-step-
induced mode of the q model take over the result? (It does not: both runs are taken over at t ~ 250 by the static
grid-scale mode of the shortest axial wave, see compare_nyquist.py.) Reads q_kink_dt0.5 and var_kink_dt0.5 beside this file (or in
$STUDY_RUNS) and writes

- timerun_kink.png: kinetic energy en_U against t (log scale) and its local growth rate 0.5 d(ln en_U)/dt, with the
  growth rates of the eigenvalues of the one-step map at the same dt (kink, and the fastest spurious pair of the q
  model). The mode that takes over is identified in compare_nyquist.py;
- timerun_kink.md: fitted growth rates.

Run: python compare_timerun.py
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from struphy import Output

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("STUDY_RUNS", HERE))
EIGS = Path(os.environ.get("STUDY_EIGS", RUNS))
C = {"q": "#2a78d6", "var": "#eb6834"}
LABEL = {"q": "q model", "var": "variational, L2"}
DT = 0.5
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#52514e",
                     "font.size": 10})


def eig(model):
    """(kink growth rate, fastest growing pair: gamma, omega) of the one-step map at DT, if the eigenvalues exist."""
    f = EIGS / f"eig_{model}_kink_dt{DT}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    g, w = np.array(d["gamma"]), np.array(d["omega"])
    real = (np.abs(w) < 1e-6) & (g > 1e-6)
    pair = (np.abs(w) >= 1e-6) & (g > 1e-6)
    kink = g[real].max() if real.any() else 0.0
    k = np.argmax(np.where(pair, g, -np.inf))
    return kink, (g[k], abs(w[k])) if pair.any() else (0.0, 0.0)


def local_rate(t, e, window=10.0):
    """0.5 d(ln e)/dt from a linear fit over a sliding window (in time units)."""
    le = np.log(e)
    n = max(3, int(round(window / (t[1] - t[0]))))
    tc, g = [], []
    for i in range(0, len(t) - n, max(1, n // 4)):
        p = np.polyfit(t[i : i + n], le[i : i + n], 1)
        tc.append(t[i : i + n].mean())
        g.append(0.5 * p[0])
    return np.array(tc), np.array(g)


runs = {m: RUNS / f"{m}_kink_dt{DT}" for m in ("q", "var")}
runs = {m: f for m, f in runs.items() if (f / "run_metadata.json").exists()}
fig = plt.figure(figsize=(12, 4.4))
gs = fig.add_gridspec(1, 2, wspace=0.25)
ax_e, ax_g = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
lines = ["| model | dt | growth rate fitted over the last 100 time units | kink eigenvalue | fastest spurious pair "
         "(gamma, omega) |", "|---|---|---|---|---|"]
for j, (m, folder) in enumerate(runs.items()):
    out = Output(str(folder))
    sc = out.scalars
    t = np.asarray(sc["t"])
    e = np.asarray(sc["en_U"]).ravel()
    ok = e > 0
    t, e = t[ok], e[ok]
    ax_e.semilogy(t, e, color=C[m], lw=1.6, label=LABEL[m])
    tc, g = local_rate(t, e)
    ax_g.plot(tc, g, color=C[m], lw=1.8, label=f"{LABEL[m]}: 0.5 d ln(en_U)/dt")
    late = t > t[-1] - 100
    g_fit = 0.5 * np.polyfit(t[late], np.log(e[late]), 1)[0]
    ev = eig(m)
    if ev is not None:
        kink, (gs_, ws_) = ev
        ax_g.axhline(kink, color=C[m], ls="--", lw=1.2, label=f"{LABEL[m]}: kink eigenvalue {kink:.4f}")
        if gs_ > 1e-4:
            ax_g.axhline(gs_, color=C[m], ls=":", lw=1.4,
                         label=f"{LABEL[m]}: spurious pair {gs_:.4f} (omega dt = {ws_ * DT:.2f})")
        lines.append(f"| {m} | {DT} | {g_fit:.5f} | {kink:.5f} | {gs_:.4f}, {ws_:.3f} |")
    else:
        lines.append(f"| {m} | {DT} | {g_fit:.5f} | - | - |")
ax_e.set_xlabel("t")
ax_e.set_ylabel("en_U (kinetic energy)")
ax_e.legend(frameon=False, fontsize=8)
ax_e.grid(color="#e4e3df")
ax_g.set_xlabel("t")
ax_g.set_ylabel("growth rate")
ax_g.legend(frameon=False, fontsize=7, loc="center left")
ax_g.annotate("static grid-scale mode (1, 6):\nq 0.1820, variational 0.1842", (330, 0.183), xytext=(150, 0.15),
              fontsize=8, color="#52514e", arrowprops=dict(arrowstyle="->", color="#52514e", lw=0.8))
ax_g.grid(color="#e4e3df")
fig.suptitle(f"Kink (q = 0.7 -> 1.3), time runs at dt = {DT}, polar cylinder 16 x 12 x 12, helical seed (1, -1)",
             x=0.01, ha="left", fontsize=11)
fig.savefig(RUNS / "timerun_kink.png", dpi=150, bbox_inches="tight")
open(RUNS / "timerun_kink.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
