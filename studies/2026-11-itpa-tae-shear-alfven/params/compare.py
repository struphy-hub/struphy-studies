"""Compare the runs of the ITPA TAE study (ShearAlfven): frequency of the TAE, spectra, (r, omega) maps, mode structure
at the TAE frequency and energy.

The radial velocity u_r (component 0 in the normalized basis) is Fourier-transformed in the poloidal and toroidal
angles (m = 9 ... 12, logical n = -1, i.e. the physical n = 6) and in time, see struphy.post_processing.mode_analysis.
The frequency of the TAE is the peak of the m = 10 spectrum at r = 0.5.

Usage: python compare.py [run names ...]  (default: all runs with run_metadata.json in $STUDY_RUNS, default the folder
of this script). Writes compare_*.png, omega_r_<run>.png, profiles_<run>.png and compare.md into $STUDY_RUNS.
"""

import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output
from struphy.post_processing import mode_analysis as ma

RUNS = Path(os.environ.get("STUDY_RUNS", Path(__file__).resolve().parent))
names = sys.argv[1:] or sorted(p.parent.name for p in RUNS.glob("*/run_metadata.json"))

MS = (9, 10, 11, 12)
M_TAE, N_LOGICAL, R_C = 10, -1, 0.5
# ideal MHD frequency of the ITPA benchmark (Koenies et al. 2018), mean of CAS3D (4.01e5) and GYGLES (4.13e5), rad/s
OMEGA_ITPA = 4.07e5
LINESTYLES = {"hcurl": "-", "hdiv": "--"}
GRIDS = ["24x44x8", "24x88x8"]


def expected_frequency(out):
    """omega_TAE = n / (2m + 1) v_A(r_c) / R0 at the center of the gap (flux-surface averages of |B0| and n0)."""
    equil, params = out.equil, out.domain.params
    q0, q1, a, R0 = (equil.params[k] for k in ("q0", "q1", "a", "R0"))
    n = abs(N_LOGICAL) * params["tor_period"]
    q_c = (2 * M_TAE + 1) / (2 * n)
    r_c = a * np.sqrt((q_c - q0) / (q1 - q0))
    eta1 = np.array([(r_c - params["a1"]) / (params["a2"] - params["a1"])])
    eta2, eta3 = (
        np.linspace(0, 1, 64, endpoint=False),
        np.linspace(0, 1, 8, endpoint=False),
    )
    abs_b = np.mean(equil.absB0(eta1, eta2, eta3))
    density = np.mean(equil.n0(eta1, eta2, eta3))
    return n / (2 * M_TAE + 1) * abs_b / np.sqrt(density) / R0


rows = []
expected = None
fig_s, ax_s = plt.subplots(figsize=(8, 5))
fig_e, ax_e = plt.subplots(figsize=(8, 4.5))
fig_p, axes_p = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for k, name in enumerate(names):
    out = Output(RUNS / name)
    expected = expected_frequency(out)
    coeffs = ma.mode_coefficients(
        out, "mhd/velocity", ms=MS, n=N_LOGICAL, component=0, representation="norm"
    )
    spectrum = ma.time_spectrum(coeffs)
    omega = ma.peak_frequency(spectrum, m=M_TAE, r=R_C, omega_max=3 * expected)
    profiles = ma.filtered_profile(spectrum, omega=omega)
    omega_si = omega / float(out.units.t)

    # colors per grid, line styles per space (H(curl) solid, H(div) dashed)
    style = {
        "color": f"C{GRIDS.index(name.split('_')[-1]) if name.split('_')[-1] in GRIDS else k}",
        "ls": LINESTYLES.get(name.split("_")[0], ":"),
    }
    i_c = int(np.argmin(np.abs(spectrum["r"].values - R_C)))
    a = spectrum.sel(m=M_TAE).values[:, i_c]
    ax_s.semilogy(
        spectrum["omega"].values,
        (a / a.max()) ** 2,
        **style,
        label=f"{name}: {omega:.4g}",
    )

    # (r, omega) map with the cylindrical continua, and the mode structure at the TAE frequency
    for fig_name, plot in (
        ("omega_r", ma.plot_omega_r(spectrum, out=out, expected=expected)),
        (
            "profiles",
            ma.plot_profiles(
                profiles,
                title=f"{name}: |u_r| harmonics at omega = {omega:.4g}",
                mark_r=R_C,
            ),
        ),
    ):
        plot.fig.savefig(RUNS / f"{fig_name}_{name}.png", dpi=120)
        plt.close(plot.fig)
    r = profiles["r"].values
    norm = float(profiles.max())
    for ax, m in zip(axes_p, (M_TAE, M_TAE + 1)):
        ax.plot(r, profiles.sel(m=m).values / norm, **style, label=name)

    en = out.scalars["en_tot"]
    t = en["t"].values if "t" in en.coords else en[en.dims[0]].values
    ax_e.plot(t, en.values / en.values[0], **style, label=name)
    rows.append(
        (
            name,
            omega,
            (omega - expected) / expected,
            omega_si,
            (omega_si - OMEGA_ITPA) / OMEGA_ITPA,
            float(en.values[-1] / en.values[0]),
        )
    )

ax_s.axvline(expected, color="k", ls="--", label=f"expected {expected:.4g}")
ax_s.axvspan(0.9 * expected, 1.1 * expected, color="grey", alpha=0.2)
ax_s.set_xlim(0, 3 * expected)
ax_s.set_ylim(1e-5, 2)
ax_s.set_xlabel(r"$\omega$ (Alfven units)")
ax_s.set_ylabel(f"power of m = {M_TAE} at r = {R_C} (normalized)")
ax_s.legend(fontsize=8)
fig_s.tight_layout()
fig_s.savefig(RUNS / "compare_spectra.png", dpi=120)

ax_e.set_xlabel("t")
ax_e.set_ylabel("total energy / initial")
ax_e.legend(fontsize=8)
fig_e.tight_layout()
fig_e.savefig(RUNS / "compare_energy.png", dpi=120)

for ax, m in zip(axes_p, (M_TAE, M_TAE + 1)):
    ax.set_title(f"|u_r| harmonic m = {m}, n = 6 at the measured frequency")
    ax.set_xlabel("minor radius r")
    ax.axvline(R_C, color="k", ls=":")
axes_p[0].legend(fontsize=8)
fig_p.tight_layout()
fig_p.savefig(RUNS / "compare_profiles.png", dpi=120)

lines = [
    f"expected omega_TAE = {expected:.4g} (center of the gap), ITPA ideal MHD {OMEGA_ITPA:.3g} rad/s",
    "",
    "| run | omega | dev. from expected | omega [rad/s] | dev. from ITPA | energy at the end / initial |",
    "|---|---|---|---|---|---|",
]
lines += [
    f"| {n} | {w:.4g} | {d:+.1%} | {s:.3g} | {ds:+.1%} | {e:.3f} |"
    for n, w, d, s, ds, e in rows
]
(RUNS / "compare.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
