"""Figure for the note: quadratic and cubic periodic B-splines on a uniform grid, the alternating ("shortest") wave in
both spaces, and the shortest wave for an odd number of cells. Also prints the numbers quoted in the note."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline


def periodic_basis(p, N):
    def f(x, j):  # B-spline j of degree p on the knots j, ..., j + p + 1, wrapped periodically (period N)
        b = BSpline.basis_element(np.arange(j, j + p + 2, dtype=float), extrapolate=False)
        return sum(np.nan_to_num(b(x + s)) for s in (-N, 0, N))

    return f


N = 8
x = np.linspace(0, N, 4001)
C = {2: "#2a78d6", 3: "#eb6834"}
fig, ax = plt.subplots(3, 1, figsize=(8.5, 8.2), sharex=True)
for p in (2, 3):
    B = periodic_basis(p, N)
    for j in (1, 2):
        ax[0].plot(x, B(x, j), color=C[p], lw=2 if j == 2 else 1, ls="-" if j == 2 else "--",
                   label=f"degree {p} (centre at {j + (p + 1) / 2:g})" if j == 2 else None)
    alt = sum((-1) ** j * B(x, j) for j in range(N))
    ax[1].plot(x, alt, color=C[p], lw=2, label=f"degree {p}" + (" (quadratic)" if p == 2 else " (cubic)"))
    k = (N - 1) / 2 / (N - 1)  # for the odd-N panel below
for kn in range(N + 1):
    for a in ax:
        a.axvline(kn, color="#b5b4af", lw=0.6, zorder=0)
ax[0].set_title("B-splines on a uniform grid (knots = grey lines): quadratic ones are centred at cell midpoints,"
                " cubic ones at knots", fontsize=9, loc="left")
ax[0].legend(frameon=False, fontsize=8)
ax[1].set_title("The alternating wave (coefficients +1, -1, +1, ...), N = 8 cells: the quadratic one is zero at every"
                " knot and odd about it,\nthe cubic one is even about every knot", fontsize=9, loc="left")
ax[1].legend(frameon=True, fontsize=8, loc="upper center", ncol=2, framealpha=1, edgecolor="none"); ax[1].set_ylim(-0.6, 0.8)
ax[1].axhline(0, color="#52514e", lw=0.6)
# odd N: shortest wave n = (N-1)/2
No = 7
xo = np.linspace(0, No, 4001)
B2 = periodic_basis(2, No)
n = (No - 1) // 2
q = sum(np.cos(2 * np.pi * n * j / No) * B2(xo, j) for j in range(No))
ax[2].plot(xo, q, color=C[2], lw=2, label=f"degree 2, N = {No} cells, shortest wave n = {n}: coefficients cos(2 pi n j / N)")
ax[2].plot(np.arange(No + 1), [sum(np.cos(2 * np.pi * n * j / No) * B2(np.array([kk + 0.0]), j)[0] for j in range(No))
                               for kk in range(No + 1)], "o", color=C[2])
ax[2].axhline(0, color="#52514e", lw=0.6)
ax[2].set_title(f"Odd number of cells: no alternating wave exists; the shortest wave does not vanish at the knots (dots)",
                fontsize=9, loc="left")
ax[2].legend(frameon=True, fontsize=8, loc="upper center", framealpha=1, edgecolor="none"); ax[2].set_ylim(-0.6, 0.8)
ax[2].set_xlabel("z / h")
fig.tight_layout()
fig.savefig(os.path.join(os.environ.get("STUDY_RUNS", "."), "splines_parity.png"), dpi=150)

# numbers: alternating quadratic at knots, and its L2 product with cubic B-splines
B2, B3 = periodic_basis(2, N), periodic_basis(3, N)
alt2 = sum((-1) ** j * B2(x, j) for j in range(N))
print("max |alt quadratic at knots| =", max(abs(sum((-1) ** j * B2(np.array([kk + 0.0]), j)[0] for j in range(N)))
                                         for kk in range(N)))
print("max |int alt quadratic * B3_j| =", max(abs(np.trapezoid(alt2 * B3(x, j), x)) for j in range(N)))
