"""Compare the mode structure (u_r, normalized basis) of two runs of the low-mode TAE study at the middle and last
saved times: radial profiles of the (m, n=-1) harmonics, the poloidal plane and relative differences.

Usage: python compare_structure.py <run_ref> <run> [out_dir]
"""

import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output

# the output folders of the runs: by default next to this script (see run_case.sh), else $STUDY_RUNS
RUNS = Path(os.environ.get("STUDY_RUNS", Path(__file__).resolve().parent))
ref_name, run_name = sys.argv[1], sys.argv[2]
out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else RUNS / f"structure_{ref_name}_vs_{run_name}"
out_dir.mkdir(parents=True, exist_ok=True)

N_LOGICAL = -1
MS = (0, 1, 2, 3)
eta1 = (np.arange(48) + 0.5) / 48
eta2 = np.linspace(0, 1, 64, endpoint=False)
eta3 = np.linspace(0, 1, 16, endpoint=False)


def radius(run):
    p = run.domain.params
    a1, a2 = (p["a1"], p["a2"]) if "a1" in p else (0.0, p["a"])
    return a1 + (a2 - a1) * eta1


def load(name):
    run = Output(RUNS / name)
    snaps = run.standard_snapshots()
    steps = {"mid": snaps[len(snaps) // 2], "end": snaps[-1]}
    out = {}
    for key, n in steps.items():
        u = run.evaluate("mhd/velocity", representation="norm", eta1=eta1, eta2=eta2, eta3=eta3, t=n, component=0)
        u = np.asarray(u.squeeze().transpose("eta1", "eta2", "eta3").values)
        out[key] = (float(run.time[n]), u)
    return run, out


def harmonics(u):
    c = np.fft.fft2(u, axes=(1, 2)) / (eta2.size * eta3.size)
    return np.stack([c[:, m, N_LOGICAL % eta3.size] for m in MS], axis=-1)  # (r, m), complex


run_ref, data_ref = load(ref_name)
run_cmp, data_cmp = load(run_name)
r = radius(run_ref)
assert np.allclose(r, radius(run_cmp)), "The runs need the same radial map (domain) for a point-wise comparison."

lines = [f"# Mode structure: {run_name} vs {ref_name} (u_r, normalized basis)", ""]
lines += [
    "| time | rel. L2 diff of u_r | rel. L2 diff, phase-aligned harmonics | per harmonic (phase-aligned) m=0,1,2,3 |",
    "|---|---|---|---|",
]
for key in ("mid", "end"):
    t_ref, u_ref = data_ref[key]
    t_cmp, u_cmp = data_cmp[key]
    assert abs(t_ref - t_cmp) < 1e-8, (t_ref, t_cmp)
    diff = np.linalg.norm(u_cmp - u_ref) / np.linalg.norm(u_ref)
    c_ref, c_cmp = harmonics(u_ref), harmonics(u_cmp)
    # one complex factor per harmonic aligns the phase (and amplitude) of the run to the reference
    per_m = []
    c_al = np.zeros_like(c_cmp)
    for k in range(len(MS)):
        num = np.vdot(c_cmp[:, k], c_ref[:, k])
        phase = num / abs(num) if abs(num) > 0 else 1.0
        c_al[:, k] = c_cmp[:, k] * phase
        per_m.append(np.linalg.norm(c_al[:, k] - c_ref[:, k]) / max(np.linalg.norm(c_ref[:, k]), 1e-300))
    diff_al = np.linalg.norm(c_al - c_ref) / np.linalg.norm(c_ref)
    lines.append(
        f"| t = {t_ref:.0f} ({key}) | {diff:.3f} | {diff_al:.3f} | " + ", ".join(f"{d:.3f}" for d in per_m) + " |"
    )

    # profiles |c_m(r)|
    fig, ax = plt.subplots(figsize=(6.5, 4))
    scale = np.abs(c_ref).max()
    for k, m in enumerate(MS):
        ax.plot(r, np.abs(c_ref[:, k]) / scale, color=f"C{k}", label=f"m = {m} ({ref_name})")
        ax.plot(r, np.abs(c_cmp[:, k]) / scale, color=f"C{k}", ls="--", label=f"m = {m} ({run_name})")
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("minor radius r")
    ax.set_ylabel(r"$|c_m(r)|$ (normalized to the reference)")
    ax.set_title(f"u_r harmonics (n = 1) at t = {t_ref:.0f}")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"profiles_{key}.png", dpi=120)
    plt.close(fig)

    # poloidal plane eta3 = 0
    R0 = 5.0
    th = 2 * np.pi * eta2
    RR = R0 + r[:, None] * np.cos(th[None, :])
    ZZ = r[:, None] * np.sin(th[None, :])
    vmax = np.abs(u_ref[:, :, 0]).max()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    for ax, field, title in (
        (axes[0], u_ref[:, :, 0], ref_name),
        (axes[1], u_cmp[:, :, 0], run_name),
        (axes[2], u_cmp[:, :, 0] - u_ref[:, :, 0], f"{run_name} - {ref_name}"),
    ):
        im = ax.pcolormesh(RR, ZZ, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("R")
    axes[0].set_ylabel("Z")
    fig.colorbar(im, ax=axes, shrink=0.8, label="u_r")
    fig.suptitle(f"u_r in the poloidal plane, t = {t_ref:.0f} (same color scale)")
    fig.savefig(out_dir / f"plane_{key}.png", dpi=120)
    plt.close(fig)

(out_dir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"figures in {out_dir}")
