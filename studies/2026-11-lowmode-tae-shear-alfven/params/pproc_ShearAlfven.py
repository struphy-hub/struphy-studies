"""Post-process and plot the low-mode TAE run (ShearAlfven).

Run as ``python pproc_ShearAlfven.py [path_out] [flags of params_ShearAlfven.py] [--show]``. The standard
plots of the examples (slices in the physical planes and 3D views of the fields) are made at the first, middle and
last saved time, together with the energy budget and the (m, n) harmonics of the radial components. Vector fields
are shown by their components in the basis of their initial condition (``given_in_basis``, else their FEEC form,
see ``Output.field_basis``), as functions of the physical coordinates; the 3D views show the scalar fields and the
first components (radial eta1 in a logical basis). The measured TAE frequency is compared with the analytical one
and saved; a deviation of more than 10% fails with an ``AssertionError`` after the figures are saved (not with
``--test-run``). Figures are written into ``<path_out>/results``. The energy and harmonics plots use plasma-plots
(``pip install "struphy[pproc]"``).

Unless ``--no-frames``, the frames of three videos of the radial velocity u_r over the run (at most ``--max-frames``
saved time steps) are written into ``<path_out>/results/frames/``: ``u_r_rz`` (the poloidal plane), ``u_r_3d_cut``
(3D, the camera on the poloidal cross-section of the cutaway) and ``u_r_3d`` (3D, the whole torus with the
cutaway), see ``struphy.post_processing.frames``. Make a video e.g. with
``ffmpeg -framerate 25 -i frame_%04d.png -pix_fmt yuv420p u_r_rz.mp4`` in a frames folder.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output
from struphy.post_processing.frames import write_videos
from struphy.post_processing.plot_3d import plot_field_3d
from struphy.post_processing.slices import plot_physical_slices

SCRIPT_DIR = Path(__file__).resolve().parent

# The flags of the run (parsed from the same command line), for the default output folder.
sys.path.insert(0, str(SCRIPT_DIR))
from params_ShearAlfven import args, parser  # noqa: E402

DEFAULT_OUTPUT = Path(args.out_folder) / args.sim_folder

# standard plots: (key, label, species, variable, vector) per field
FIELDS = [
    ("u", "U", "mhd", "velocity", True),
    ("b", "B", "em_fields", "b_field", True),
]
# (m, n) harmonics of the radial (eta1) components
HARMONICS = [("u_r", "mhd/velocity"), ("b_r", "em_fields/b_field")]
ENERGIES = ("en_U", "en_B")  # energy scalars shown next to en_tot

# the TAE: toroidal mode number n (the logical mode N_LOGICAL along eta3 times tor_period of the domain) in the gap
# between the poloidal mode numbers M and M + 1, see the description of the params file
M_TAE = 1
N_LOGICAL = -1
# relative tolerance of the measured TAE frequency: the formula gives the center of the gap, whose relative width
# is about the inverse aspect ratio r_c/R0 = 0.1 at the gap
TAE_RTOL = 0.1


def tae_frequency(run, basis):
    """Measured and expected frequency of the TAE.

    Expected: omega_TAE = n / (2m + 1) * v_A(r_c) / R0 at the radius r_c of the gap, q(r_c) = (2m + 1) / (2n), for the
    q profile q0 + (q1 - q0) r^2/a^2 of the AdhocTorus; v_A = |B| / sqrt(n) (in units of the Alfven velocity) with the
    flux-surface averages of |B| and of the number density. Measured: the dominant frequency of the (M_TAE, n)
    harmonic of the radial velocity on the flux surface r_c over the run, projected on its initial phase so that a
    standing or a propagating mode gives the same frequency.
    """
    equil, params = run.equil, run.domain.params
    q0, q1, a, R0 = (equil.params[k] for k in ("q0", "q1", "a", "R0"))
    n = abs(N_LOGICAL) * params["tor_period"]
    q_c = (2 * M_TAE + 1) / (2 * n)
    r_c = a * np.sqrt((q_c - q0) / (q1 - q0))
    r_min, r_max = radial_range(run)
    eta1 = np.array([(r_c - r_min) / (r_max - r_min)])
    eta2, eta3 = np.linspace(0, 1, 64, endpoint=False), np.linspace(0, 1, 8, endpoint=False)
    abs_b = np.mean(equil.absB0(eta1, eta2, eta3))
    density = np.mean(equil.n0(eta1, eta2, eta3))
    expected = n / (2 * M_TAE + 1) * abs_b / np.sqrt(density) / R0

    # the radial component is the first one of a logical basis (the FEEC form for Cartesian initial conditions)
    basis = run.field_form("mhd/velocity") if basis == "physical" else basis
    u_r = run.evaluate("mhd/velocity", representation=basis, eta1=eta1, eta2=eta2, eta3=eta3, component=0)
    modes = u_r.squeeze("eta1", drop=True).plasma.analysis.mode_spectrum()
    coefficient = modes.sel(m=M_TAE, n=N_LOGICAL)
    initial = complex(coefficient.isel(t=0))
    # projected on the initial phase; without initial amplitude in this harmonic its real part
    phase = initial / abs(initial) if abs(initial) > 0 else 1.0
    signal = (coefficient * np.conj(phase)).real
    # a frequency needs a time series (not e.g. the single saved step of a one-step test)
    if signal.sizes["t"] < 3:
        measured = float("nan")
    else:
        measured = float(signal.plasma.analysis.spectral_peaks(n_peaks=1).omega_refined[0])
    print(
        f"TAE (m = {M_TAE}, {M_TAE + 1}, n = {n}) at r_c = {r_c:.3g}: omega = {measured:.4g}, expected {expected:.4g}"
    )
    return measured, expected


def radial_range(run):
    """The minor radius at eta1 = 0 and eta1 = 1: (a1, a2) of the HollowTorus, (0, a) of the IGAPolarTorus."""
    params = run.domain.params
    if "a1" in params:
        return params["a1"], params["a2"]
    return 0.0, params["a"]


def minor_radius(run):
    """The minor radius as a function of eta1."""
    a1, a2 = radial_range(run)
    return lambda eta1: a1 + (a2 - a1) * eta1


def continuum(run, r, m, n):
    """Cylindrical shear Alfven continuum omega_m(r) = |n q(r) - m| / (q(r) R0) * v_A(r) of the poloidal mode number m.

    v_A = |B| / sqrt(n) with the flux-surface averages of |B| and of the number density (in units of the Alfven
    velocity), as for the expected TAE frequency."""
    equil = run.equil
    q0, q1, a, R0 = (equil.params[k] for k in ("q0", "q1", "a", "R0"))
    q = q0 + (q1 - q0) * r**2 / a**2
    r_min, r_max = radial_range(run)
    eta1 = (r - r_min) / (r_max - r_min)
    eta2, eta3 = np.linspace(0, 1, 32, endpoint=False), np.array([0.0])
    abs_b = np.mean(equil.absB0(eta1, eta2, eta3), axis=(1, 2))
    density = np.mean(equil.n0(eta1, eta2, eta3), axis=(1, 2))
    return np.abs(n * q - m) / (q * R0) * abs_b / np.sqrt(density)


def frequency_diagnostics(run, expected, measured, results_dir, num_r=48, pad=8):
    """Time spectra of the (m, n) harmonics of the radial velocity u_r (normalized basis: the physical radial
    component, non-zero on the axis for m = 1) at all radii.

    The Fourier coefficients c_m(r, t) of u_r = sum_m c_m(r, t) exp(2 pi i (m eta2 + n eta3)) (n = N_LOGICAL) are
    transformed in time with a Hann window (zero-padded ``pad`` times, which interpolates the spectrum but does not
    improve the resolution 2 pi / T); the power is summed over +omega and -omega (standing and propagating waves alike).
    Figures: ``tae_spectrum`` (power of the harmonics at r_c and integrated over r, with the expected frequency and
    the 10% band), ``tae_omega_r`` (power over (r, omega) per harmonic, with the cylindrical continua) and
    ``tae_profiles`` (|c_m(r)| at the measured frequency, i.e. the mode structure of the TAE). Returns the frequency
    of the peak of the m = M_TAE spectrum at r_c."""
    n_tor = abs(N_LOGICAL) * run.domain.params["tor_period"]
    r_min, r_max = radial_range(run)
    q0, q1, a = (run.equil.params[k] for k in ("q0", "q1", "a"))
    r_c = a * np.sqrt(((2 * M_TAE + 1) / (2 * n_tor) - q0) / (q1 - q0))
    # cell-centred in eta1: avoids the pole (inf in the normalized basis) and the walls
    eta1 = (np.arange(num_r) + 0.5) / num_r
    eta2, eta3 = np.linspace(0, 1, 32, endpoint=False), np.linspace(0, 1, 8, endpoint=False)
    u_r = run.evaluate("mhd/velocity", representation="norm", eta1=eta1, eta2=eta2, eta3=eta3, component=0)
    u_r = u_r.transpose("t", "eta1", "eta2", "eta3")
    t = u_r["t"].values
    r = r_min + (r_max - r_min) * eta1
    # Fourier coefficients c[t, r, m, n] (numpy FFT index m, n mod size)
    coef = np.fft.fft2(u_r.values, axes=(2, 3)) / (eta2.size * eta3.size)
    ms = (0, 1, 2, 3)
    c = np.stack([coef[:, :, m, N_LOGICAL % eta3.size] for m in ms], axis=-1)  # (t, r, m)

    nt = t.size
    if nt < 4:
        return float("nan")
    dt = t[1] - t[0]
    window = np.hanning(nt)[:, None, None]
    nfft = pad * nt
    spec = np.fft.fft((c - c.mean(axis=0)) * window, n=nfft, axis=0)
    omega = 2 * np.pi * np.fft.fftfreq(nfft, d=dt)
    pos = omega >= 0
    w = omega[pos]
    # |F(omega)|^2 + |F(-omega)|^2
    neg = (-np.arange(nfft)) % nfft
    power = (np.abs(spec) ** 2 + np.abs(spec[neg]) ** 2)[pos]  # (omega, r, m)
    amp = np.sqrt(power)

    i_c = int(np.argmin(np.abs(r - r_c)))
    m_idx = ms.index(M_TAE)
    w_max = 3 * expected
    sel = w <= w_max
    peak = float(w[sel][np.argmax(power[sel, i_c, m_idx][1:]) + 1])
    resolution = 2 * np.pi / (t[-1] - t[0])
    np.save(os.path.join(results_dir, "omega_tae_fft.npy"), peak)
    print(f"FFT peak of the m = {M_TAE} harmonic at r_c: omega = {peak:.4g} (resolution 2 pi/T = {resolution:.2g})")

    # (A) spectra at r_c and integrated over r
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, m in enumerate(ms):
        if m in (M_TAE, M_TAE + 1):
            ax.semilogy(w[sel], power[sel, i_c, k] / power[sel, i_c, m_idx].max(), label=f"m = {m} at r = {r[i_c]:.2f}")
    total = power[sel].sum(axis=(1, 2))
    ax.semilogy(w[sel], total / total.max(), "k", lw=1, label="all m, all r")
    ax.axvspan((1 - TAE_RTOL) * expected, (1 + TAE_RTOL) * expected, color="grey", alpha=0.2, label="expected ±10%")
    ax.axvline(expected, color="k", ls="--", label=f"expected $\\omega_{{TAE}}$ = {expected:.4g}")
    ax.axvline(measured, color="C3", ls=":", label=f"measured (spectral_peaks) = {measured:.4g}")
    ax.axvline(peak, color="C2", ls="-.", label=f"FFT peak = {peak:.4g}")
    ax.set_xlim(0, w_max)
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel(r"$\omega$  [$v_A / x_{unit}$]")
    ax.set_ylabel("power (normalized)")
    ax.set_title(f"Time spectrum of u_r harmonics, n = {n_tor}  (T = {t[-1]:.0f}, $2\\pi/T$ = {resolution:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs = {"tae_spectrum": fig}

    # (B) omega-r maps with the cylindrical continua
    r_fine = np.linspace(max(r_min, 1e-3), r_max, 200)
    fig, axes = plt.subplots(1, len(ms), figsize=(4 * len(ms), 4.5), sharey=True)
    vmax = amp[sel].max()
    for k, (m, ax) in enumerate(zip(ms, axes)):
        im = ax.pcolormesh(r, w[sel], np.log10(amp[sel, :, k] / vmax + 1e-12), vmin=-3, vmax=0, shading="auto")
        for mc in range(0, 5):
            ax.plot(r_fine, continuum(run, r_fine, mc, n_tor), "w-" if mc == m else "w:", lw=1)
        ax.axhline(expected, color="r", ls="--", lw=1)
        ax.axvline(r_c, color="r", ls=":", lw=1)
        ax.set_ylim(0, w_max)
        ax.set_xlabel("minor radius r")
        ax.set_title(f"m = {m}")
    axes[0].set_ylabel(r"$\omega$")
    fig.colorbar(im, ax=axes, label="log10 amplitude (normalized)")
    fig.suptitle("u_r harmonics over (r, omega); white: cylindrical continua (solid: same m), red: expected TAE")
    figs["tae_omega_r"] = fig

    # (C) radial profiles at the measured frequency
    i_w = int(np.argmin(np.abs(w - peak)))
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, m in enumerate(ms):
        ax.plot(r, amp[i_w, :, k] / amp[i_w].max(), label=f"m = {m}")
    ax.axvline(r_c, color="k", ls=":", lw=1)
    ax.set_xlabel("minor radius r")
    ax.set_ylabel(r"$|\hat c_m(r, \omega)|$ (normalized)")
    ax.set_title(f"Mode structure of u_r at $\\omega$ = {peak:.4g} (n = {n_tor})")
    ax.legend()
    fig.tight_layout()
    figs["tae_profiles"] = fig
    np.savez(os.path.join(results_dir, "tae_spectra.npz"), omega=w[sel], r=r, ms=ms, amp=amp[sel], t=t)
    finish(figs, results_dir, True)
    return peak


def finish(figs, results_dir, save_figs):
    """Save the figures to ``results_dir`` and close them (``save_figs``), or keep them open for ``plt.show()``."""
    if save_figs:
        for fig_name, fig in figs.items():
            fig.savefig(os.path.join(results_dir, f"{fig_name}.png"))
            plt.close(fig)


def main(path_out=DEFAULT_OUTPUT, save_figs=True, frames=True, max_frames=500):
    run = Output(path_out)
    snapshots = run.standard_snapshots()
    # VTK files (for the 3D views) of the standard snapshots only
    run.pproc(create_vtk=True, physical=True, vtk_steps=snapshots)
    times = [float(run.time[n]) for n in snapshots]

    # The profiling packaging picks these files up from here and uploads them with the run.
    results_dir = os.path.join(run.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    # the basis of the components of each vector field (None for a scalar field)
    bases = {f"{s}/{v}": run.field_basis(f"{s}/{v}") if vector else None for _, _, s, v, vector in FIELDS}

    # the frequency of the TAE, compared with the analytical value at the end (after the figures)
    measured, expected = tae_frequency(run, bases["mhd/velocity"])
    np.save(os.path.join(results_dir, "omega_tae.npy"), measured)
    np.save(os.path.join(results_dir, "omega_tae_expected.npy"), expected)
    frequency_diagnostics(run, expected, measured, results_dir)

    finish({"energies": run.plot.energies(parts=ENERGIES).fig}, results_dir, save_figs)
    for n, t in zip(snapshots, times):
        # one field at a time, saved right away, so that few figures are open at once
        for key, label, species, variable, _ in FIELDS:
            name = f"{species}/{variable}"
            field = (key, f"{label} at t = {t:.3g}", name, n, None, bases[name])
            slices = plot_physical_slices(run, run.domain, run.grid.num_elements, [field], components=True)
            # e.g. u1_slices_rz_t1.5 for the first component of U in the R-z plane at t = 1.5
            finish({f"{fig_name}_t{t:.6g}": fig for fig_name, fig in slices.items()}, results_dir, save_figs)

        figs = {}
        for key, name in HARMONICS:
            # the logical radial component: in the basis of the initial condition unless it is Cartesian
            basis = bases[name] if bases[name] != "physical" else run.field_form(name)
            data = run.evaluate(name, representation=basis, t=n, component=0)
            data.name = key
            result = data.plasma.plot.mode_profiles(
                scale=(1, run.domain.params["tor_period"]),
                x_of=minor_radius(run),
                xlabel="minor radius r",
                top=4,
            )
            figs[f"{key}_harmonics_t{t:.6g}"] = result.fig
        finish(figs, results_dir, save_figs)

        # 3D views of the scalar fields and of the first (radial eta1, in a logical basis) components, saved as
        # <field>_3d_t<time>.png and <field>1_3d_t<time>.png
        for key, _, species, variable, _ in FIELDS:
            basis = bases[f"{species}/{variable}"]
            if basis is None:
                filename = os.path.join(results_dir, f"{key}_3d_t{t:.6g}.png")
                plot_field_3d(run.path_out, variable, species=species, snapshot=n, filename=filename)
            elif basis in (run.field_form(f"{species}/{variable}"), "physical"):
                # the VTK files hold the components in the FEEC form and the Cartesian ones
                filename = os.path.join(results_dir, f"{key}1_3d_t{t:.6g}.png")
                plot_field_3d(
                    run.path_out,
                    variable,
                    species=species,
                    snapshot=n,
                    component=0,
                    physical=basis == "physical",
                    filename=filename,
                )
            else:
                print(f"No 3D view of {species}/{variable}: its basis {basis!r} is not in the VTK files.")

    if frames:
        # frames of videos of the radial velocity u_r (the first component in the normalized basis, radial for the
        # circular flux surfaces of the HollowTorus): the poloidal plane and two 3D views, at most max_frames saved
        # time steps, in <path_out>/results/frames/u_r_rz, u_r_3d_cut and u_r_3d (frame_*.png)
        videos = write_videos(
            run,
            "mhd/velocity",
            os.path.join(results_dir, "frames"),
            component=0,
            basis="norm",
            prefix="u_r",
            label="radial velocity u_r",
            max_frames=max_frames,
        )
        for video, files in videos.items():
            if files:  # (empty on the ranks other than 0)
                print(f"{len(files)} frames of {video} in {os.path.join(results_dir, 'frames', video)}")

    if not save_figs:
        plt.show()

    # not for the few steps of --test-run
    if not args.test_run:
        assert abs(measured - expected) <= TAE_RTOL * expected, (
            f"The TAE frequency {measured:.4g} differs from the expected {expected:.4g} by more than {TAE_RTOL:.0%}."
        )


if __name__ == "__main__":
    # the flags of the run are parsed by params_ShearAlfven.py, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process and plot a saved simulation run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: the output folder of the run flags, sim_data beside this script)",
    )
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_parser.add_argument("--no-frames", action="store_true", help="Do not write the frames of the videos.")
    pproc_parser.add_argument(
        "--max-frames", type=int, default=500, help="Largest number of frames per video (default: 500)."
    )
    pproc_args = pproc_parser.parse_args(other_flags)
    main(
        pproc_args.path_out,
        save_figs=not pproc_args.show,
        frames=not pproc_args.no_frames,
        max_frames=pproc_args.max_frames,
    )
