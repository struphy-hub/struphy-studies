"""Mode structure of the eigenvectors of the one-step map (eig_block.py --modes 3), read from
eig_<model>_<case>_dt<dt>[_r32]_modes.npz beside this file (or in $STUDY_RUNS):

- modes_axis.png: kink, dt = 0.25: the fastest-growing mode with the commuting projection (the axis mode), the kink
  of the same scheme, and the fastest-growing mode with the L2 transport (the kink).
- modes_kink_suydam.png: kink (dt = 0.25) and Suydam mode (dt = 0.125) of the q model and the variational model in
  the plane z = 0, and the radial profiles of their poloidal harmonic (Suydam also with 32 radial elements).
- modes_torus.png: kink of the polar torus (AdhocTorus, dt = 0.125) of both models in the plane phi = 0, and the
  radial profiles of the poloidal harmonics m = 0..3 of the radial velocity.

The plane pictures show the radial velocity (color, normalized by its maximum) and the in-plane velocity (arrows); the
profiles are normalized by the maximum of the in-plane velocity; the phase of each mode is chosen such that the dominant harmonic of u_r is real at its peak.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("STUDY_RUNS", HERE))
C = {"q": "#2a78d6", "var": "#eb6834", "var_pi": "#1baf7a"}
LABEL = {"q": "q model", "var": "variational, L2", "var_pi": "variational, commuting"}
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#52514e",
                     "font.size": 10})
RES = {"kink": 0.71, "suydam": 0.71, "torus": 0.41}  # radius of the resonant surface (q = 1, 2, 1)


def load(tag):
    f = RUNS / f"eig_{tag}_modes.npz"
    return dict(np.load(f)) if f.exists() else None


def mode(d, k, m_dom=None):
    """Cylindrical components (u_r, u_theta, u_out) of mode k as complex fields on (eta1, eta2), the in-plane
    coordinates (a, b), the radius and the dominant poloidal harmonic."""
    X, Y, Z = d["xyz"]
    u = d["u"][k]
    torus = np.ptp(Z) > 1e-8
    if torus:  # plane phi = 0 of the torus: (R, Z)
        R = np.hypot(X, Y)
        a, b = R, Z
        ua, ub = (u[0] * X + u[1] * Y) / R, u[2]
        uo = (-u[0] * Y + u[1] * X) / R
    else:
        a, b = X, Y
        ua, ub, uo = u[0], u[1], u[2]
    a0, b0 = a[0].mean(), b[0].mean()
    th = np.arctan2(b - b0, a - a0)
    ur = ua * np.cos(th) + ub * np.sin(th)
    ut = -ua * np.sin(th) + ub * np.cos(th)
    r = np.hypot(a - a0, b - b0).mean(axis=1)
    F = np.fft.fft(ur, axis=1) / ur.shape[1]
    if m_dom is None:
        m_dom = int(np.argmax(np.abs(F).sum(axis=0)))
    i = np.argmax(np.abs(F[:, m_dom]))
    ph = np.exp(-1j * np.angle(F[i, m_dom]))  # phase: harmonic m_dom of u_r real at its peak
    ur, ut, uo, ua, ub = (f * ph for f in (ur, ut, uo, ua, ub))
    nrm = np.max(np.hypot(ua.real, ub.real))
    return dict(a=a, b=b, r=r, ur=ur / nrm, ut=ut / nrm, uo=uo / nrm, ua=ua.real / nrm, ub=ub.real / nrm, m=m_dom,
                gamma=float(d["gamma"][k]), torus=torus)


def harm(f, m):
    return np.abs(np.fft.fft(f, axis=1)[:, m]) / f.shape[1]


def wrap(f):
    """Close the periodic poloidal direction for plotting."""
    return np.concatenate([f, f[:, :1]], axis=1)


def plane(ax, md, title, res=None, zoom=None):
    """u_r (color, normalized by its own maximum) and the in-plane velocity (arrows, about uniformly spaced); with
    zoom, only the disc r < zoom around the axis."""
    a0, b0 = md["a"][0].mean(), md["b"][0].mean()
    ur = md["ur"].real
    nr = md["r"].size
    rows = np.arange(nr) if zoom is None else np.arange(nr)[md["r"] <= zoom * 1.02]
    vmax = np.abs(ur[rows]).max()
    pc = ax.pcolormesh(wrap(md["a"][rows]), wrap(md["b"][rows]), wrap(ur[rows]) / vmax, cmap="RdBu_r", vmin=-1,
                       vmax=1, shading="gouraud")
    rmax = md["r"][rows[-1]]
    dr = rmax / 11
    pts = []
    for i in rows[:: max(1, len(rows) // 11)][1:]:
        k = max(4, int(round(2 * np.pi * md["r"][i] / dr)))
        pts += [(i, j) for j in np.unique((np.arange(k) * ur.shape[1] / k).astype(int))]
    I, J = np.array(pts).T
    ua, ub = md["ua"][I, J], md["ub"][I, J]
    sc = np.hypot(md["ua"][rows], md["ub"][rows]).max()
    ax.quiver(md["a"][I, J], md["b"][I, J], ua / sc, ub / sc, color="#0b0b0b", scale=1 / (0.9 * dr),
              scale_units="xy", angles="xy", width=0.006, headwidth=4)
    t = np.linspace(0, 2 * np.pi, 200)
    if res is not None and (zoom is None or res < zoom):
        ax.plot(a0 + res * np.cos(t), b0 + res * np.sin(t), color="#52514e", ls="--", lw=1)
    ax.plot(a0 + rmax * np.cos(t), b0 + rmax * np.sin(t), color="#b5b4af", lw=0.8)
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=9, loc="left")
    return pc


def pick(d, target=None):
    """Index of the fastest mode, or of the mode with growth rate closest to target."""
    g = d["gamma"]
    return int(np.argmax(g)) if target is None else int(np.argmin(np.abs(g - target)))


# ---------------------------------------------------------------- axis mode: commuting vs L2 (kink, dt = 0.25)
dpi, dl2 = load("var_pi_kink_dt0.25"), load("var_kink_dt0.25")
if dpi is not None and dl2 is not None:
    m_ax = mode(dpi, pick(dpi), 1)
    g_kink = dl2["gamma"][pick(dl2)]
    m_pk = mode(dpi, pick(dpi, g_kink), 1)
    m_l2 = mode(dl2, pick(dl2), 1)
    fig = plt.figure(figsize=(14, 3.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.4], wspace=0.12)
    for j, (md, t, z) in enumerate(((m_ax, f"commuting: fastest mode\ngamma = {m_ax['gamma']:.4f}", None),
                                    (m_ax, "commuting: fastest mode,\nzoom r < 0.15", 0.15),
                                    (m_pk, f"commuting: kink\ngamma = {m_pk['gamma']:.4f}", None),
                                    (m_l2, f"L2 transport: fastest mode\n= kink, gamma = {m_l2['gamma']:.4f}", None))):
        pc = plane(fig.add_subplot(gs[j]), md, t, RES["kink"], zoom=z)
    ax = fig.add_subplot(gs[4])
    for md, col, ls, lab in ((m_ax, C["var_pi"], "-", "commuting, fastest"), (m_pk, C["var_pi"], "--", "commuting, kink"),
                             (m_l2, C["var"], "-", "L2, fastest (kink)")):
        h = harm(md["ur"], 1)
        ax.plot(md["r"], h / h.max(), color=col, ls=ls, lw=2, label=lab)
    ax.axvline(RES["kink"], color="#52514e", ls="--", lw=1)
    ax.set_xlabel("r")
    ax.set_title("|u_r|, harmonic m = 1 (normalized)", fontsize=9, loc="left")
    ax.set_xlim(0, 1)
    ax.grid(color="#e4e3df")
    ax.legend(frameon=False, fontsize=8)
    fig.colorbar(pc, ax=fig.axes[:4], shrink=0.4, pad=0.04, location="bottom", label="u_r / max |u_r|")
    fig.suptitle("Kink (1, -1), dt = 0.25, variational model: commuting projection vs L2 transport "
                 "(dashed circle: q = 1)", x=0.01, ha="left", fontsize=11)
    fig.savefig(RUNS / "modes_axis.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------- kink and Suydam: q vs variational
rows = [("kink", "dt0.25", 1, "Kink (1, -1), dt = 0.25"), ("suydam", "dt0.125", 2, "Suydam (2, -1), dt = 0.125")]
if all(load(f"{m}_{c}_{dt}") is not None for c, dt, _, _ in rows for m in ("q", "var")):
    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.35, 1.35], wspace=0.22, hspace=0.32)
    for i, (case, dt, m, title) in enumerate(rows):
        if case == "kink":
            axs = [fig.add_subplot(gs[i, 2:])]
        else:
            axs = [fig.add_subplot(gs[i, 2]), fig.add_subplot(gs[i, 3])]
        for j, model in enumerate(("q", "var")):
            d = load(f"{model}_{case}_{dt}")
            md = mode(d, pick(d), m)
            pc = plane(fig.add_subplot(gs[i, j]), md, f"{title}\n{LABEL[model]}, gamma = {md['gamma']:.5f}", RES[case])
            if case == "kink":
                for f, ls, nm in (("ur", "-", "u_r"), ("ut", "--", "u_theta"), ("uo", ":", "u_z")):
                    axs[0].plot(md["r"], harm(md[f], m), color=C[model], ls=ls, lw=2 if f == "ur" else 1.4,
                                label=f"{LABEL[model]}: {nm}")
                axs[0].set_title("Kink: harmonic m = 1 of the velocity components", fontsize=9, loc="left")
            else:
                d32 = load(f"{model}_{case}_{dt}_r32")
                m32 = mode(d32, pick(d32), m) if d32 is not None else None
                for ax, f, nm in zip(axs, ("ur", "ut"), ("u_r", "u_theta")):
                    ax.plot(md["r"], harm(md[f], m), color=C[model], lw=2,
                            label=f"{LABEL[model]}, 16 radial (gamma = {md['gamma']:.5f})")
                    if m32 is not None:
                        ax.plot(m32["r"], harm(m32[f], m), color=C[model], lw=1.2, ls="--",
                                label=f"{LABEL[model]}, 32 radial (gamma = {m32['gamma']:.5f})")
                    ax.set_title(f"Suydam: harmonic m = 2 of {nm}", fontsize=9, loc="left")
        for ax in axs:
            ax.axvline(RES[case], color="#52514e", ls="--", lw=1)
            ax.set_xlim(0, 1)
            ax.set_xlabel("r")
            ax.grid(color="#e4e3df")
            ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.colorbar(pc, ax=fig.axes, shrink=0.4, pad=0.02, location="left", label="u_r / max |u_r|")
    fig.suptitle("Fastest-growing mode, polar cylinder 16 x 12 x 12: q model vs variational (L2 transport); "
                 "dashed circle / line: resonant surface", x=0.01, ha="left", fontsize=11)
    fig.savefig(RUNS / "modes_kink_suydam.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------- polar torus: q vs variational
dq, dv = load("q_torus_dt0.125"), load("var_torus_dt0.125")
if dq is not None and dv is not None:
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.2)
    ax = fig.add_subplot(gs[2])
    styles = {0: ":", 1: "-", 2: "--", 3: "-."}
    for j, (model, d) in enumerate((("q", dq), ("var", dv))):
        md = mode(d, pick(d), 1)
        pc = plane(fig.add_subplot(gs[j]), md, f"{LABEL[model]}, gamma = {md['gamma']:.5f}", RES["torus"])
        n = md["ur"].shape[1]
        for mm, ls in styles.items():
            h = np.abs(np.fft.fft(md["ur"], axis=1)[:, [mm, -mm]]).max(axis=1) / n if mm else harm(md["ur"], 0)
            ax.plot(md["r"], h, color=C[model], ls=ls, lw=2 if mm == 1 else 1.3, label=f"{LABEL[model]}: |m| = {mm}")
    ax.set_ylim(bottom=0)
    ax.axvline(RES["torus"], color="#52514e", ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("r (distance from the axis)")
    ax.set_title("|u_r| of the poloidal harmonics", fontsize=9, loc="left")
    ax.grid(color="#e4e3df")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.colorbar(pc, ax=fig.axes[1:], shrink=0.6, pad=0.04, location="bottom", label="u_r / max |u_r|")
    fig.suptitle("Polar torus (AdhocTorus, R0 = 3), n = -1, dt = 0.125: kink in the plane phi = 0 (dashed: q = 1)",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(RUNS / "modes_torus.png", dpi=150, bbox_inches="tight")
print("done")
