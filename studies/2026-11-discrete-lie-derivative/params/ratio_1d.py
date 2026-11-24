"""1D bending ratio r(u_h) = ||d/dz P u_h|| / ||d/dz u_h|| on a periodic uniform grid of N cells (length 1).

u_h: quadratic periodic spline whose coefficients are the wave cos(2 pi n j / N) (the velocity, degree p-1 = 2);
P: L2 projection onto cubic periodic splines (the V^1 space, degree p = 3), or interpolation at the knots (Pi^1).
r = 1 means the discrete bending equals the exact bending of the same spline velocity; r = 0 is a spurious kernel.
Run: python ratio_1d.py   (prints a table for N = 12, 13, 24, 25, 48, 49)
"""

import numpy as np
from scipy.interpolate import BSpline


def periodic_basis(N, p, x):
    """Values (and derivatives) of the N periodic B-splines of degree p on [0, 1) at points x."""
    h = 1.0 / N
    V = np.zeros((len(x), N))
    D = np.zeros((len(x), N))
    for j in range(N):
        knots = (j + np.arange(p + 2)) * h
        b = BSpline.basis_element(knots, extrapolate=False)
        for s in (-1.0, 0.0, 1.0):
            v = b(x + s)
            d = b.derivative()(x + s)
            V[:, j] += np.nan_to_num(v)
            D[:, j] += np.nan_to_num(d)
    return V, D


def ratios(N, nq=40):
    g, w = np.polynomial.legendre.leggauss(nq)
    h = 1.0 / N
    x = np.concatenate([(c + 0.5 * (g + 1)) * h for c in range(N)])
    wq = np.tile(0.5 * w * h, N)
    V2, D2 = periodic_basis(N, 2, x)
    V3, D3 = periodic_basis(N, 3, x)
    M3 = V3.T @ (wq[:, None] * V3)
    # interpolation at the knots: cubic B-spline j has its centre at knot j + 2
    knots = (np.arange(N) * h) % 1.0
    I3, _ = periodic_basis(N, 3, knots)
    I2, _ = periodic_basis(N, 2, knots)
    out = []
    for n in range(1, N // 2 + 1):
        c = np.cos(2 * np.pi * n * np.arange(N) / N)
        du = D2 @ c
        exact = np.sqrt(np.sum(wq * du**2))
        e_l2 = np.linalg.solve(M3, V3.T @ (wq * (V2 @ c)))
        e_pi = np.linalg.solve(I3, I2 @ c)
        r_l2 = np.sqrt(np.sum(wq * (D3 @ e_l2) ** 2)) / exact
        r_pi = np.sqrt(np.sum(wq * (D3 @ e_pi) ** 2)) / exact
        out.append((n, r_pi, r_l2))
    return out


if __name__ == "__main__":
    for N in (12, 13, 24, 25, 48, 49):
        rs = ratios(N)
        print(f"N = {N:3d}: shortest wave n = {rs[-1][0]:3d}: r(Pi^1) = {rs[-1][1]:.3e}, r(L2) = {rs[-1][2]:.3e};"
              f"  n = N/4: r(L2) = {rs[N // 4 - 1][2]:.3f};  min over n: r(L2) = {min(r[2] for r in rs):.3e}")
