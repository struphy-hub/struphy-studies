"""Figures of the study on the discrete Lie derivative L^h_u B0,h = curl P(B0,h x u), from the tables of the kernel
checks (beside this file or in $STUDY_RUNS):

- compare_coupling.png: (left) the discrete bending |curl P(B0 x u)| / |u| of smooth velocity waves (m = 1, axial wave
  number n) in each H(div) component on the kink screw pinch, 16 x 12 x 12 (nyquist_kernel_z12_m1-3-5-6.md, from
  check_nyquist_kernel.py); (right) the 1D ratio r = |d/dz P u_h| / |d/dz u_h| of the shortest wave against the
  number of cells N, even and odd (ratio_1d.py);
- compare_kernel_K.png: the spurious bending s_h of the parallel flow z = cos(2 pi n eta3) B0 against k, N3 = 12 and
  13, with the sound frequency c_s k and k B_theta (kernel_K_polar_z12/13.md, kernel_K_slab_z12.md, from
  check_kernel_K.py).

Run: python compare_lie.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ratio_1d import ratios

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("STUDY_RUNS", HERE))
C = {"q": "#2a78d6", "var": "#eb6834"}
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#52514e",
                     "font.size": 10})


def table(name):
    rows = []
    for line in open(RUNS / name):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) < 3 or cells[0] in ("component", "n") or set(cells[0]) <= set("-"):
            continue
        rows.append(cells)
    return rows


# ---------------------------------------------------------------- coupling of smooth waves + 1D ratio
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
rows = [r for r in table("nyquist_kernel_z12_m1-3-5-6.md") if r[1] == "1"]
for comp, ls in (("u^1 (radial flux)", "-"), ("u^2 (poloidal flux)", "--"), ("u^3 (axial flux)", ":")):
    rr = [r for r in rows if r[0] == comp]
    n = [int(r[2]) for r in rr]
    for col, key, lab in ((3, "q", "Pi^1 (q model)"), (4, "var", "L2 (variational)")):
        y = np.maximum([float(r[col]) for r in rr], 1e-16)
        ax[0].semilogy(n, y, ls, color=C[key], marker="o", ms=3, label=f"{comp}, {lab}")
ax[0].axvline(6, color="#888", lw=0.8)
ax[0].text(5.8, 1e-9, "n = N3/2 = 6\n(alternating wave)", ha="right", fontsize=8, color="#555")
ax[0].set(xlabel="axial wave number n of the velocity (m = 1)", ylabel="|curl P(B0 x u)| / |u|",
          title="Discrete bending of smooth waves, kink screw pinch 16 x 12 x 12", ylim=(1e-16, 50))
ax[0].legend(fontsize=7, loc="lower left")

Ns = np.arange(8, 61)
r_pi = np.array([ratios(N)[-1][1] for N in Ns])
r_l2 = np.array([ratios(N)[-1][2] for N in Ns])
ev, od = Ns % 2 == 0, Ns % 2 == 1
ax[1].semilogy(Ns[od], r_l2[od], "o-", color=C["var"], ms=3, label="L2, odd N")
ax[1].semilogy(Ns[od], r_pi[od], "o-", color=C["q"], ms=3, label="Pi^1, odd N")
ax[1].semilogy(Ns[od], 7.0 / Ns[od], "k:", lw=1, label="7 / N")
ax[1].semilogy(Ns[ev], np.maximum(r_l2[ev], 1e-16), "x", color=C["var"], label="L2, even N (round-off)")
ax[1].semilogy(Ns[ev], np.maximum(r_pi[ev], 1e-16), "+", color=C["q"], label="Pi^1, even N (round-off)")
ax[1].set(xlabel="number of cells N", ylabel="r = |d/dz P u_h| / |d/dz u_h|",
          title="1D model: bending ratio r of the shortest wave", ylim=(1e-17, 2))
ax[1].legend(fontsize=7, loc="center right")
fig.tight_layout()
fig.savefig(RUNS / "compare_coupling.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------- condition (K): parallel flows
fig, ax = plt.subplots(figsize=(6.5, 4.3))
for nz, mk in ((12, "o"), (13, "s")):
    rr = table(f"kernel_K_polar_z{nz}.md")
    k = np.array([float(r[1]) for r in rr])
    ax.semilogy(k[1:], [float(r[2]) for r in rr][1:], mk + "-", color=C["var"], ms=4,
                label=f"variational (L2), N3 = {nz}")
    ax.semilogy(k[1:], [float(r[3]) for r in rr][1:], mk + "--", color=C["q"], ms=4, label=f"q model (Pi^1), N3 = {nz}")
    if nz == 12:
        ax.semilogy(k[1:], [float(r[6]) for r in rr][1:], "k-", lw=1, label="c_s k (sound, r = 0.5)")
        ax.semilogy(k[1:], [float(r[7]) for r in rr][1:], "k:", lw=1, label="k B_theta (r = 0.5)")
ax.set(xlabel="axial wave number k = 2 pi n / Lz", ylabel="s_h = |curl P(B0,h x z_h)| / |z_h|",
       title="Spurious bending of the parallel flow z = cos(k z) B0\n(kink screw pinch; aligned slab: exactly 0)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(RUNS / "compare_kernel_K.png", dpi=150, bbox_inches="tight")
print("wrote compare_coupling.png, compare_kernel_K.png in", RUNS)
