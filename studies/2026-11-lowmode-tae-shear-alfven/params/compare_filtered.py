"""Compare the mode structure at the TAE frequency (time-Fourier amplitude of the u_r harmonics, tae_spectra.npz)
and at other frequency bands, between runs. Usage: python compare_filtered.py ref run [run ...]"""

import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# the output folders of the runs: by default next to this script (see run_case.sh), else $STUDY_RUNS
RUNS = Path(os.environ.get("STUDY_RUNS", Path(__file__).resolve().parent))
names = sys.argv[1:]
ref = names[0]
out = RUNS / f"structure_filtered_{'_'.join(names)}"
out.mkdir(exist_ok=True)

data = {}
for n in names:
    d = np.load(RUNS / n / "results" / "tae_spectra.npz")
    peak = float(np.load(RUNS / n / "results" / "omega_tae_fft.npy"))
    data[n] = (d["omega"], d["r"], list(d["ms"]), d["amp"], peak)

w0, r0, ms, amp0, peak0 = data[ref]
bands = {"TAE peak": None, "continuum band 0.035-0.055": (0.035, 0.055)}
lines = ["| run | omega peak | band | rel. L2 diff of |c_m(r)| (m=0..3, normalized) | per m |", "|---|---|---|---|---|"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, (band, lim) in zip(axes, bands.items()):

    def profile(n):
        w, r, _, amp, peak = data[n]
        if lim is None:
            P = amp[int(np.argmin(np.abs(w - peak)))]
        else:
            sel = (w >= lim[0]) & (w <= lim[1])
            P = np.sqrt((amp[sel] ** 2).sum(axis=0))
        return r, P

    r_ref, P_ref = profile(ref)
    scale = P_ref.max()
    for i, n in enumerate(names):
        r, P = profile(n)
        if n != ref:
            # onto the radii of the reference (runs with another hole or polar splines have other radii at the same eta1)
            P = np.stack([np.interp(r_ref, r, P[:, k], left=np.nan, right=np.nan) for k in range(P.shape[1])], axis=1)
            r = r_ref
        P_n = P / np.nanmax(P) * 1.0
        for k, m in enumerate(ms[:3]):
            ax.plot(r, P_n[:, k], color=f"C{k}", ls=["-", "--", ":", "-."][i % 4], label=f"m = {m} ({n})")
        if n != ref:
            Pr = P_ref / scale
            ok = np.all(np.isfinite(P_n), axis=1)  # common radii
            d = np.linalg.norm(P_n[ok] - Pr[ok]) / np.linalg.norm(Pr[ok])
            per = [
                np.linalg.norm(P_n[ok, k] - Pr[ok, k]) / max(np.linalg.norm(Pr[ok, k]), 1e-300) for k in range(len(ms))
            ]
            lines.append(
                f"| {n} vs {ref} | {data[n][4]:.4g} vs {peak0:.4g} | {band} | {d:.3f} | "
                + ", ".join(f"{x:.3f}" for x in per)
                + " |"
            )
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_title(f"|c_m(r, omega)|, {band}")
    ax.set_xlabel("minor radius r")
    ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(out / "filtered_profiles.png", dpi=120)
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(out)
