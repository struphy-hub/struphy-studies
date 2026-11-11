"""Compare the runs of the low-mode TAE study: table of frequencies, spectra at r_c and mode profiles at omega_TAE.

Usage: python compare.py [run names ...]  (default: all runs next to this script with a results/tae_spectra.npz)
Writes compare_*.png and compare.md into the folder of the runs ($STUDY_RUNS, default: next to this script).
"""

import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# the output folders of the runs: by default next to this script (see run_case.sh), else $STUDY_RUNS
RUNS = Path(os.environ.get("STUDY_RUNS", Path(__file__).resolve().parent))
names = sys.argv[1:] or sorted(p.parent.parent.name for p in RUNS.glob("*/results/tae_spectra.npz"))

rows = []
fig_s, ax_s = plt.subplots(figsize=(8, 5))
fig_p, axes_p = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
expected = None
for k, name in enumerate(names):
    res = RUNS / name / "results"
    expected = float(np.load(res / "omega_tae_expected.npy"))
    meas = float(np.load(res / "omega_tae.npy"))
    fft = float(np.load(res / "omega_tae_fft.npy"))
    d = np.load(res / "tae_spectra.npz")
    w, r, ms, amp = d["omega"], d["r"], list(d["ms"]), d["amp"]
    r_c = 0.5
    i_c = int(np.argmin(np.abs(r - r_c)))
    p = amp[:, i_c, ms.index(1)] ** 2
    ax_s.semilogy(w, p / p.max(), color=f"C{k}", label=name)
    i_w = int(np.argmin(np.abs(w - fft)))
    norm = amp[i_w].max()
    for ax, m in zip(axes_p, (1, 2)):
        ax.plot(r, amp[i_w, :, ms.index(m)] / norm, color=f"C{k}", label=name)
    # fraction of the m = 1 amplitude inside r < 0.25 (support towards the axis)
    prof1 = amp[i_w, :, ms.index(1)]
    inner = np.trapezoid(prof1[r < 0.25], r[r < 0.25]) / np.trapezoid(prof1, r)
    rows.append((name, meas, fft, (fft - expected) / expected, inner))

ax_s.axvline(expected, color="k", ls="--", label=f"expected {expected:.4g}")
ax_s.axvspan(0.9 * expected, 1.1 * expected, color="grey", alpha=0.2)
ax_s.set_xlim(0, 3 * expected)
ax_s.set_ylim(1e-5, 2)
ax_s.set_xlabel(r"$\omega$")
ax_s.set_ylabel("power of m = 1 at r_c (normalized)")
ax_s.legend(fontsize=8)
fig_s.tight_layout()
fig_s.savefig(RUNS / "compare_spectra.png", dpi=120)
for ax, m in zip(axes_p, (1, 2)):
    ax.set_title(f"|u_r| harmonic m = {m} at the measured frequency")
    ax.set_xlabel("minor radius r")
    ax.axvline(0.5, color="k", ls=":")
axes_p[0].legend(fontsize=8)
fig_p.tight_layout()
fig_p.savefig(RUNS / "compare_profiles.png", dpi=120)

lines = [
    f"expected omega_TAE = {expected:.4g}",
    "",
    "| run | omega (spectral_peaks) | omega (FFT peak) | rel. dev. (FFT) | m=1 weight r<0.25 |",
    "|---|---|---|---|---|",
]
lines += [f"| {n} | {a:.4g} | {b:.4g} | {c:+.1%} | {e:.2f} |" for n, a, b, c, e in rows]
(RUNS / "compare.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
