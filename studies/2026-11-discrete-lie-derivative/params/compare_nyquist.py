"""Grid-scale axial mode: the block (m, n) = (1, n_max) of the shortest axial wave (n_max = N3 // 2) of the kink case,
dt = 0.5, for N3 = 8, 11, 12, 13, 16, 24 axial and 16, 32 radial elements (run_nyquist.sh; files
nyq_<model>_r<N1>_z<N3>[_mid] beside this file or in $STUDY_RUNS). Writes

- nyquist.png: the growth rate of the static mode against N3 (even and odd); the mode of N3 = 12 for both models: u_z
  in the plane z = 0 (a knot, where the in-plane velocity of this mode vanishes) and the in-plane velocity in the plane
  half an element further (where u_z vanishes); the radial profiles of their m = 1 harmonic;
- nyquist.md: the growth rates.
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
C = {"q": "#2a78d6", "var": "#eb6834"}
LABEL = {"q": "q model", "var": "variational, L2"}
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#52514e",
                     "font.size": 10})


def rate(model, nr, nz):
    f = RUNS / f"nyq_{model}_r{nr}_z{nz}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    g, w = np.array(d["gamma"]), np.array(d["omega"])
    static = (np.abs(w) < 1e-6) & (g > 1e-6)
    return (g[static].max() if static.any() else 0.0), d["mn"], d["leak"]


def mode(model, suffix):
    f = RUNS / f"nyq_{model}_r16_z12{suffix}_modes.npz"
    if not f.exists():
        return None
    d = np.load(f)
    u = d["u"][int(np.argmax(d["gamma"]))]
    X, Y, _ = d["xyz"]
    th = np.arctan2(Y, X)
    ur = u[0] * np.cos(th) + u[1] * np.sin(th)
    ut = -u[0] * np.sin(th) + u[1] * np.cos(th)
    # the shortest axial wave is real: one phase; pick the sign/phase with the largest real part
    ph = np.exp(-1j * np.angle(u.ravel()[np.argmax(np.abs(u.ravel()))]))
    return dict(X=X, Y=Y, r=np.hypot(X, Y).mean(axis=1), ur=(ur * ph).real, ut=(ut * ph).real, ux=(u[0] * ph).real,
                uy=(u[1] * ph).real, uz=(u[2] * ph).real)


def h1(f):
    return np.abs(np.fft.fft(f, axis=1)[:, 1]) / f.shape[1]


lines = ["| model | elements (radial x poloidal x axial) | block | growth rate of the static mode | leak |",
         "|---|---|---|---|---|"]
fig = plt.figure(figsize=(15, 4.3))
gs = fig.add_gridspec(1, 5, width_ratios=[1.15, 1, 1, 1, 1.25], wspace=0.25)
ax = fig.add_subplot(gs[0])
for m in ("q", "var"):
    ev, od = [], []
    for nz in (8, 11, 12, 13, 16, 24):
        r = rate(m, 16, nz)
        if r is None:
            continue
        (ev if nz % 2 == 0 else od).append((nz, r[0]))
        lines.append(f"| {m} | 16 x 12 x {nz} | ({r[1][0]}, {r[1][1]}) | {r[0]:.4f} | {r[2]:.0e} |")
    r = rate(m, 32, 12)
    if r is not None:
        lines.append(f"| {m} | 32 x 12 x 12 | ({r[1][0]}, {r[1][1]}) | {r[0]:.4f} | {r[2]:.0e} |")
        ax.plot([12], [r[0]], "D", color=C[m], ms=8, mfc="none", label=f"{LABEL[m]}, 32 radial")
    if ev:
        ev = np.array(ev)
        ax.plot(ev[:, 0], ev[:, 1], "o-", color=C[m], lw=2, label=f"{LABEL[m]}, even N3")
    if od:
        od = np.array(od)
        ax.plot(od[:, 0], od[:, 1], "x", color=C[m], ms=9, mew=2, label=f"{LABEL[m]}, odd N3")
ax.set_xlabel("axial elements N3 (block n = N3 // 2)")
ax.set_ylabel("growth rate of the static mode")
ax.set_xticks([8, 12, 16, 24])
ax.annotate("odd N3 = 11, 13:\nno static mode", (12, 0.0), xytext=(13.5, 0.06), fontsize=8, color="#52514e",
            arrowprops=dict(arrowstyle="->", color="#52514e", lw=0.8))
ax.set_ylim(-0.01, 0.34)
ax.grid(color="#e4e3df")
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.set_title("Kink case, shortest axial wave, dt = 0.5", fontsize=9, loc="left")

axp = fig.add_subplot(gs[4])
k = (np.arange(16) * 96 / 16).astype(int)
for j, m in enumerate(("q", "var")):
    knot, mid = mode(m, ""), mode(m, "_mid")
    if knot is not None:
        a = fig.add_subplot(gs[1 + j])
        f = knot["uz"] / np.abs(knot["uz"]).max()
        a.pcolormesh(np.c_[knot["X"], knot["X"][:, :1]], np.c_[knot["Y"], knot["Y"][:, :1]], np.c_[f, f[:, :1]],
                     cmap="RdBu_r", vmin=-1, vmax=1, shading="gouraud")
        a.set_aspect(1)
        a.axis("off")
        a.set_title(f"{LABEL[m]}: u_z at z = 0\n(normalized, red > 0 > blue)", fontsize=9, loc="left")
        nrm = np.abs(knot["uz"]).max()
        axp.plot(knot["r"], h1(knot["uz"]) / h1(knot["uz"]).max(), color=C[m], ls=":", lw=2, label=f"{LABEL[m]}: u_z")
    if mid is not None and m == "q":
        a = fig.add_subplot(gs[3])
        pts = [(i, j) for i in range(5, 64, 6)
               for j in np.unique((np.arange(max(6, round(28 * mid["r"][i]))) * 96 / max(6, round(28 * mid["r"][i])))
                                  .astype(int))]
        I, J = np.array(pts).T
        sc = np.hypot(mid["ux"], mid["uy"]).max()
        a.quiver(mid["X"][I, J], mid["Y"][I, J], mid["ux"][I, J] / sc, mid["uy"][I, J] / sc, color="#0b0b0b",
                 scale=1 / 0.09, scale_units="xy", angles="xy", width=0.006)
        t = np.linspace(0, 2 * np.pi, 200)
        a.plot(np.cos(t), np.sin(t), color="#b5b4af", lw=0.8)
        a.set_aspect(1)
        a.axis("off")
        a.set_title(f"{LABEL[m]}: in-plane velocity\nhalf an element further in z", fontsize=9, loc="left")
    if mid is not None:
        axp.plot(mid["r"], h1(mid["ur"]) / h1(mid["ur"]).max(), color=C[m], ls="-", lw=2,
                 label=f"{LABEL[m]}: u_r (mid-element)")
axp.set_xlim(0, 1)
axp.set_xlabel("r")
axp.grid(color="#e4e3df")
axp.legend(frameon=False, fontsize=7)
axp.set_title("N3 = 12: harmonic m = 1 (normalized)", fontsize=9, loc="left")
fig.suptitle("Static grid-scale mode at the shortest axial wave (polar cylinder 16 x 12 x N3, kink equilibrium)", x=0.01,
             ha="left", fontsize=11)
fig.savefig(RUNS / "nyquist.png", dpi=150, bbox_inches="tight")
open(RUNS / "nyquist.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
