"""Exact eigenvalues of the one-step map of a linear model (one time step of the whole splitting) in an invariant
Fourier block, built column by column with the model's own time stepping.

- Polar cylinder: block (m, n) of the poloidal and axial mode numbers. One column per radial ring of every
  tensor-product component (input cos(2 pi (m j/N2 + n k/N3)), output amplitude (2/(N2 N3)) sum x exp(-i psi)), plus
  one column per in-plane polar pair of a 2-form (velocity and magnetic field): its m = +1 coordinate
  w = alpha_0 P_0 + alpha_1 P_1 is read off the extraction operator (alpha_l: m = +1 Fourier coefficient on ring 0
  of the theta-flux of polar basis function l).
- Polar torus: block n of the toroidal mode number (all poloidal harmonics coupled): one column per non-toroidal index
  of every segment.
The block is exact if the map commutes with the symmetry; the "leak" (the part of the output outside the block,
relative to the input) measures it. gamma = log|lambda| / dt, omega = arg(lambda) / dt.

Run (4 MPI ranks): mpirun -n 4 python eig_block.py --model var --case kink --dt 0.25 --out eig_var_kink_dt0.25.json
"""

import argparse
import json
import os
import time

import numpy as np
from maybempi import MPI

from cases import CASES, domain_and_equil, make_model, model_vectors

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["q", "var", "var_pi"], required=True)
ap.add_argument("--case", choices=list(CASES), required=True)
ap.add_argument("--dt", type=float, required=True)
ap.add_argument("--num-elements", type=int, nargs=3, default=None, help="default 16 12 12 (cylinder), 16 12 8 (torus)")
ap.add_argument("--out", required=True)
ap.add_argument("--mn", type=int, nargs=2, default=None, help="other block (m, n) than the one of the case (cylinder)")
ap.add_argument("--plane-eta3", type=float, default=0.0, help="logical eta3 of the plane of --modes")
ap.add_argument("--load-A", default=None, help="one-step matrix of an earlier run (.npy): no assembly, two columns checked")
ap.add_argument("--modes", type=int, default=0,
                help="save the velocity of the K fastest-growing modes (and of the physical one) in a poloidal plane to "
                     "<out>_modes.npz")
a = ap.parse_args()
rank = MPI.COMM_WORLD.Get_rank()

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, grids

case = CASES[a.case]
torus = case["geom"] == "torus"
ne = tuple(a.num_elements or ((16, 12, 8) if torus else (16, 12, 12)))
model = make_model(a.model)
domain, equil = domain_and_equil(a.case, ne)
here = os.path.dirname(os.path.abspath(a.out))
sim = Simulation(
    model=model,
    env=EnvironmentOptions(out_folders=os.path.join(here, "sims"), sim_folder=os.path.splitext(os.path.basename(a.out))[0]),
    time_opts=Time(dt=a.dt, Tend=a.dt, split_algo="Strang"),
    domain=domain,
    equil=equil,
    grid=grids.TensorProductGrid(num_elements=ne),
    derham_opts=DerhamOptions(degree=(3, 3, 3), bcs=(("free", "dirichlet"), None, None), polar_splines=True),
)
sim.allocate()
from struphy.propagators.base import Propagator

derham = Propagator.derham
U, B, P, RHO = model_vectors(model, a.model)
vecs = [U, B, P]
bops = [derham.boundary_ops["2"], derham.boundary_ops["2"], None]


def comps(v):
    tps = list(v.tp.blocks) if hasattr(v.tp, "blocks") else [v.tp]
    return [(v.pol[c], tps[c], v.space.n_rings[c]) for c in range(len(tps))]


layout = []  # (field, comp, kind, shape, offset) in the order of PolarVector.toarray
o = 0
for f, v in enumerate(vecs):
    for c, (pol, t, nr) in enumerate(comps(v)):
        layout.append((f, c, "pol", pol.shape, o))
        o += pol.size
        n = t.space.npts
        sh = (n[0] - nr, n[1], n[2])
        layout.append((f, c, "tp", sh, o))
        o += int(np.prod(sh))
ntot = o


def get():
    return np.concatenate([v.toarray(allreduce=True) for v in vecs])


def set_state(x):
    for f, c, kind, sh, off in layout:
        pol, t, nr = comps(vecs[f])[c]
        blk = x[off : off + int(np.prod(sh))].reshape(sh)
        if kind == "pol":
            pol[:] = blk
        else:
            st, en = t.starts, t.ends
            full = np.zeros(tuple(t.space.npts))
            full[nr:] = blk
            sl = tuple(slice(s, e + 1) for s, e in zip(st, en))
            t[sl] = full[sl]
    for v, bop in zip(vecs, bops):
        if bop is not None:
            bop.dot(v).copy(out=v)
        v.update_ghost_regions()
    RHO.__imul__(0.0)
    RHO.update_ghost_regions()


def step(x):
    """The one-step map applied to x (a unit-norm input: the solver tolerances are absolute for some propagators)."""
    nx = np.linalg.norm(x)
    set_state(x / nx)
    model.integrate(a.dt, "Strang")
    return get() * nx


mm, nn = a.mn or case["mn"]
# the in-plane polar pair of a 2-form is a vector at the axis: it carries only the poloidal harmonics m = +-1
segs_pol = [(f, sh, off) for f, c, kind, sh, off in layout if kind == "pol" and sh[0] == 2]
if not torus and abs(mm) != 1:
    segs_pol = []
assert not [sh for f, c, kind, sh, off in layout if kind == "pol" and sh[0] not in (0, 2)], "unexpected polar blocks"

if torus:
    # block n: one column per non-toroidal index of every segment
    segs = [(sh, off) for f, c, kind, sh, off in layout if int(np.prod(sh)) > 0]
    cols = [(sh, off, r) for sh, off in segs for r in range(int(np.prod(sh[:-1])))]

    def zph(n3):
        return 2 * np.pi * nn * np.arange(n3) / n3

    def unit(ic):
        sh, off, r = cols[ic]
        e = np.zeros(ntot)
        X = np.zeros(sh).reshape(-1, sh[-1])
        X[r] = np.cos(zph(sh[-1]))
        e[off : off + X.size] = X.ravel()
        return e

    def coeffs(y):
        out = []
        for sh, off in segs:
            X = y[off : off + int(np.prod(sh))].reshape(-1, sh[-1])
            out.append((X * np.exp(-1j * zph(sh[-1]))[None]).sum(axis=1) * 2.0 / sh[-1])
        return np.concatenate(out)

    def recon(cy):
        rec = np.zeros(ntot)
        o3 = 0
        for sh, off in segs:
            nr_ = int(np.prod(sh[:-1]))
            rec[off : off + int(np.prod(sh))] = np.real(cy[o3 : o3 + nr_, None] * np.exp(1j * zph(sh[-1]))[None]).ravel()
            o3 += nr_
        return rec

else:
    # block (m, n): one column per ring of every tensor-product segment, one per in-plane polar pair
    E2 = derham.extraction_ops["2"]

    def alpha(space):
        al = []
        for l in range(2):
            w = space.zeros()
            w.pol[0][l, 0] = 1.0
            t1 = E2.T.dot(w)[1]
            arr = t1.toarray()
            if MPI.COMM_WORLD.Get_size() > 1:
                arr = MPI.COMM_WORLD.allreduce(arr)
            c = arr.reshape(tuple(t1.space.npts))[0, :, 0]
            al.append(np.mean(c * np.exp(-2j * np.pi * mm * np.arange(c.size) / c.size)))
        al = np.array(al)
        return al, np.linalg.inv(np.array([al.real, al.imag]))

    ALPHA = {f: alpha(vecs[f].space) for f, sh, off in segs_pol}
    segs_tp = [(sh, off) for f, c, kind, sh, off in layout if kind == "tp"]

    def tph(sh):
        return 2 * np.pi * (mm * np.arange(sh[1])[:, None] / sh[1] + nn * np.arange(sh[2])[None, :] / sh[2])

    cols = [("pol", i) for i in range(len(segs_pol))] + [("tp", (s, r)) for s, (sh, off) in enumerate(segs_tp) for r in range(sh[0])]

    def unit(ic):
        kind, idx = cols[ic]
        e = np.zeros(ntot)
        if kind == "pol":
            f, sh, off = segs_pol[idx]
            w = np.exp(2j * np.pi * nn * np.arange(sh[1]) / sh[1])
            e[off : off + int(np.prod(sh))] = (ALPHA[f][1] @ np.array([w.real, w.imag])).ravel()
        else:
            s, r = idx
            sh, off = segs_tp[s]
            X = np.zeros(sh)
            X[r] = np.cos(tph(sh))
            e[off : off + X.size] = X.ravel()
        return e

    def coeffs(y):
        out = []
        for f, sh, off in segs_pol:
            Pp = y[off : off + int(np.prod(sh))].reshape(sh)
            al = ALPHA[f][0]
            out.append([np.mean((al[0] * Pp[0] + al[1] * Pp[1]) * np.exp(-2j * np.pi * nn * np.arange(sh[1]) / sh[1]))])
        for sh, off in segs_tp:
            X = y[off : off + int(np.prod(sh))].reshape(sh)
            out.append((X * np.exp(-1j * tph(sh))[None]).sum(axis=(1, 2)) * 2.0 / (sh[1] * sh[2]))
        return np.concatenate([np.atleast_1d(np.asarray(v)) for v in out])

    def recon(cy):
        rec = np.zeros(ntot)
        i = 0
        for f, sh, off in segs_pol:
            w = cy[i] * np.exp(2j * np.pi * nn * np.arange(sh[1]) / sh[1])
            rec[off : off + int(np.prod(sh))] = (ALPHA[f][1] @ np.array([w.real, w.imag])).ravel()
            i += 1
        for sh, off in segs_tp:
            cc = cy[i : i + sh[0]]
            rec[off : off + int(np.prod(sh))] = np.real(cc[:, None, None] * np.exp(1j * tph(sh))[None]).ravel()
            i += sh[0]
        return rec


d = len(cols)
if rank == 0:
    print(f"EIGSETUP {a.model} {a.case} dt={a.dt} ne={ne} d={d} ntot={ntot}", flush=True)
leak = 0.0
t0 = time.time()
if a.load_A:
    A = np.load(a.load_A)
    assert A.shape == (d, d), f"{a.load_A}: shape {A.shape}, block dimension {d}"
    for ic in (0, d // 2):
        e = unit(ic)
        y = step(e)
        cy = coeffs(y)
        leak = max(leak, np.linalg.norm(y - recon(cy)) / np.linalg.norm(e))
        dev = np.linalg.norm(cy - A[:, ic]) / np.linalg.norm(cy)
        if rank == 0:
            print(f"CHECK column {ic}: relative difference to the loaded matrix {dev:.1e}", flush=True)
        assert dev < 1e-5, "the loaded matrix does not belong to this setup"
else:
    A = np.zeros((d, d), dtype=complex)
    for ic in range(d):
        e = unit(ic)
        y = step(e)
        cy = coeffs(y)
        A[:, ic] = cy
        leak = max(leak, np.linalg.norm(y - recon(cy)) / np.linalg.norm(e))
        if rank == 0 and (ic % 50 == 0 or ic == d - 1):
            print(f"COL {ic}/{d} {(time.time() - t0) / (ic + 1):.2f}s/col leak={leak:.1e}", flush=True)
    if rank == 0:
        np.save(os.path.splitext(a.out)[0] + "_A.npy", A)
lam, V = np.linalg.eig(A)
keep = np.abs(lam) > 0
lam, V = lam[keep], V[:, keep]
gam = np.log(np.abs(lam)) / a.dt
omg = np.angle(lam) / a.dt
# radial peak and dominant poloidal harmonic (torus) of each eigenvector, from the first tensor-product velocity segment
sh0, off0 = [(sh, off) for f, c, kind, sh, off in layout if kind == "tp"][0]
if torus:
    o_u = sum(int(np.prod(sh[:-1])) for sh, off in segs if off < off0)
    U0 = V[o_u : o_u + sh0[0] * sh0[1]].reshape(sh0[0], sh0[1], -1)
    rad = np.abs(U0).sum(axis=1)
    F = np.abs(np.fft.fft(U0, axis=1)).sum(axis=0)
    mdom = np.argmax(F, axis=0)
    mdom = np.where(mdom <= sh0[1] // 2, mdom, mdom - sh0[1])
else:
    o_u = len(segs_pol)
    rad = np.abs(V[o_u : o_u + sh0[0]])
    mdom = np.full(lam.size, mm)
rpeak = (np.argmax(rad, axis=0) + 0.5) / sh0[0]
order = np.argsort(-gam)
if rank == 0:
    for i in order[:10]:
        print(f"EIG {a.model} {a.case} dt={a.dt}: gamma={gam[i]:+.5e} omega={omg[i]:+.4e} m={mdom[i]} r_peak~{rpeak[i]:.2f}",
              flush=True)
    json.dump(
        dict(model=a.model, case=a.case, dt=a.dt, num_elements=ne, mn=[mm, nn], d=d, leak=leak, t=time.time() - t0,
             gamma=gam.tolist(), omega=omg.tolist(), rpeak=rpeak.tolist(), m=[int(x) for x in mdom]),
        open(a.out, "w"),
    )
    print(f"DONE {a.model} {a.case} dt={a.dt} d={d} max gamma={gam.max():+.5e} leak={leak:.1e} t={time.time() - t0:.0f}s",
          flush=True)

# ---------------------------------------------------------------- mode structure in a poloidal plane (eta3 = 0)
if a.modes > 0:
    real_grow = [i for i in order if abs(omg[i]) < 1e-6 and gam[i] > 1e-6]
    sel = list(order[: a.modes]) + [i for i in real_grow[:1] if i not in order[: a.modes]]
    e1 = (np.arange(64) + 0.5) / 64
    e2 = np.arange(96) / 96
    e3 = np.array([a.plane_eta3])
    XYZ = np.asarray(domain(e1, e2, e3, squeeze_out=True))
    vel = model.mhd.velocity.spline
    Us = []
    for i in sel:
        cy = V[:, i] / V[np.argmax(np.abs(V[:, i])), i]  # phase: largest coefficient real
        uc = []
        for ph in (cy, 1j * cy):  # real and imaginary part of the complex mode (phase 0 and 90 degrees)
            set_state(recon(ph))
            u = vel(e1, e2, e3)  # collective
            uc.append(np.asarray(domain.push(u, e1, e2, e3, kind="2", squeeze_out=True)))
        Us.append(uc[0] - 1j * uc[1])
    if rank == 0:
        # u is complex: the velocity at the axial/toroidal phase psi is Re(u exp(i psi))
        np.savez(os.path.splitext(a.out)[0] + "_modes.npz", eta1=e1, eta2=e2, xyz=XYZ, u=np.array(Us),
                 gamma=gam[sel], omega=omg[sel], physical=np.array([i in real_grow[:1] for i in sel]))
        print(f"MODES saved {len(sel)}: gamma={np.round(gam[sel], 5).tolist()}", flush=True)
