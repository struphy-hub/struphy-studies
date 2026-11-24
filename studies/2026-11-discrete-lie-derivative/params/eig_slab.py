"""Discrete spectrum of the linear models in a homogeneous periodic slab (B0 = e_z), block by block in the wave vector.

Setup: periodic Cuboid 1 x 1 x 2 pi R0 (R0 = 3, the length of the cylinder of the study) with HomogenSlab (B0 = e_z,
n0 = 1, beta = 0.1, p0 = 0.05), velocity in H(div), degree 3, N1 x N2 x N3 cells. Everything is periodic and
homogeneous, so the one-step map commutes with the shifts by one cell in every direction: the Bloch block of the wave
vector (n1, n2, n3) is exactly invariant (one coefficient per component: u1, u2, u3, b1, b2, b3, p or sqrt(p), a 7 x 7
block). It is built column by column with the model's own time step (input cos(psi) in one component,
psi = 2 pi (n1 i/N1 + n2 j/N2 + n3 k/N3), output amplitude of exp(i psi)); omega = arg(lambda)/dt, gamma = log|lambda|/dt.

Exact ideal-MHD frequencies of the continuous wave vector k = 2 pi (n1/L1, n2/L2, n3/L3): shear Alfven omega = |k_z| v_A,
fast/slow omega^2 = k^2 (v_A^2 + c_s^2)/2 +- sqrt(k^4 (v_A^2 + c_s^2)^2/4 - k^2 k_z^2 v_A^2 c_s^2), v_A = 1,
c_s^2 = gamma p0/n0 (gamma = 5/3), plus one zero mode (the divergence of b).

Run: mpirun -n 4 python eig_slab.py --model var --nz 12 [--n12 1 0] [--dt 0.02]   (eig_slab_<model>_z<nz>_k<n1><n2>.json)
"""

import argparse
import json
import os
import time

import numpy as np
from maybempi import MPI

from cases import make_model, model_vectors

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["q", "var", "var_pi", "cc"], required=True)
ap.add_argument("--nz", type=int, default=12)
ap.add_argument("--nxy", type=int, nargs=2, default=[8, 8])
ap.add_argument("--n12", type=int, nargs=2, default=[1, 0], help="in-plane wave numbers (n1, n2) of the blocks")
ap.add_argument("--dt", type=float, default=0.02)
ap.add_argument("--scan", choices=["n3", "n1"], default="n3",
                help="n3: blocks (n1, n2, 0..N3/2) (along B0); n1: blocks (0..N1/2, n2, n3) with n3 = --n3 (across B0)")
ap.add_argument("--n3", type=int, default=1, help="axial wave number of the blocks of --scan n1")
a = ap.parse_args()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids  # noqa: E402

R0, BETA, GAMMA = 3.0, 0.1, 5.0 / 3.0
L = (1.0, 1.0, 2 * np.pi * R0)
ne = (a.nxy[0], a.nxy[1], a.nz)
model = make_model(a.model)
here = os.path.dirname(os.path.abspath(__file__))
tag = f"k{a.n12[0]}{a.n12[1]}" if a.scan == "n3" else f"x{a.n12[1]}{a.n3}"
out = os.path.join(os.environ.get("STUDY_RUNS", here), f"eig_slab_{a.model}_z{a.nz}_{tag}.json")
sim = Simulation(
    model=model,
    env=EnvironmentOptions(out_folders=os.path.join(here, "sims"), sim_folder=f"eig_slab_{a.model}_z{a.nz}"),
    time_opts=Time(dt=a.dt, Tend=a.dt, split_algo="Strang"),
    domain=domains.Cuboid(r1=L[0], r2=L[1], r3=L[2]),
    equil=equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0, beta=BETA, n0=1.0),
    grid=grids.TensorProductGrid(num_elements=ne, mpi_dims_mask=(True, True, False)),
    derham_opts=DerhamOptions(degree=(3, 3, 3), bcs=(None, None, None), polar_splines=False),
)
sim.allocate()

U, B, P, RHO = model_vectors(model, a.model)
vecs = [U, B, P]
NAMES = ["u1", "u2", "u3", "b1", "b2", "b3", "p" if a.model != "q" else "sqrt_p"]  # b: 1-form for cc


def blocks(v):
    return list(v.blocks) if hasattr(v, "blocks") else [v]


COMPS = [t for v in vecs for t in blocks(v)]
SHAPES = [tuple(t.space.npts) for t in COMPS]


def local(t):
    return tuple(slice(s, e + 1) for s, e in zip(t.starts, t.ends))


def get():
    """The full coefficient arrays of the 7 components (collective)."""
    res = []
    for t, sh in zip(COMPS, SHAPES):
        full = np.zeros(sh)
        sl = local(t)
        full[sl] = t[sl]
        res.append(comm.allreduce(full) if comm.Get_size() > 1 else full)
    return res


def set_state(arrs):
    for t, full in zip(COMPS, arrs):
        sl = local(t)
        t[sl] = full[sl]
    for v in vecs:
        v.update_ghost_regions()
    RHO.__imul__(0.0)
    RHO.update_ghost_regions()


def phase(sh, n):
    i, j, k = np.meshgrid(*[np.arange(s) for s in sh], indexing="ij")
    return 2 * np.pi * (n[0] * i / sh[0] + n[1] * j / sh[1] + n[2] * k / sh[2])


def block(n):
    """7 x 7 one-step map of the Bloch block n = (n1, n2, n3), and the leak out of the block."""
    real = all((2 * nd) % Nd == 0 for nd, Nd in zip(n, ne))  # self-conjugate block: the cosine is the whole harmonic
    fac = 1.0 if real else 2.0
    A = np.zeros((7, 7), dtype=complex)
    leak = 0.0
    for c in range(7):
        arrs = [np.zeros(sh) for sh in SHAPES]
        arrs[c] = np.cos(phase(SHAPES[c], n))
        nin = np.sqrt(sum(np.sum(x**2) for x in arrs))
        set_state([x / nin for x in arrs])
        model.integrate(a.dt, "Strang")
        y = [x * nin for x in get()]
        res2 = 0.0
        for r, (x, sh) in enumerate(zip(y, SHAPES)):
            ps = phase(sh, n)
            A[r, c] = np.sum(x * np.exp(-1j * ps)) * fac / x.size
            res2 += np.sum((x - np.real(A[r, c] * np.exp(1j * ps))) ** 2)  # output of cos(psi) is Re(A exp(i psi))
        leak = max(leak, np.sqrt(res2) / nin)
    return A, leak


def exact(n):
    k = np.array([2 * np.pi * nd / Ld for nd, Ld in zip(n, L)])
    k2, kz2 = np.sum(k**2), k[2] ** 2
    va2, cs2 = 1.0, GAMMA * BETA / 2
    s = k2 * (va2 + cs2) / 2
    d = np.sqrt(max(s**2 - k2 * kz2 * va2 * cs2, 0.0))
    return dict(alfven=float(np.sqrt(kz2 * va2)), fast=float(np.sqrt(s + d)), slow=float(np.sqrt(max(s - d, 0.0))))


rows = []
t0 = time.time()
blocks_n = ([(a.n12[0], a.n12[1], n3) for n3 in range(a.nz // 2 + 1)] if a.scan == "n3"
            else [(n1, a.n12[1], a.n3) for n1 in range(ne[0] // 2 + 1)])
for n in blocks_n:
    A, leak = block(n)
    lam, V = np.linalg.eig(A)
    omg, gam = np.angle(lam) / a.dt, np.log(np.abs(lam)) / a.dt
    # velocity / field / pressure content of each eigenvector
    w = np.abs(V) ** 2
    cont = dict(u=w[:3].sum(0).tolist(), b=w[3:6].sum(0).tolist(), p=w[6].tolist(), u_comp=w[:3].T.tolist())
    rows.append(dict(n=list(n), omega=omg.tolist(), gamma=gam.tolist(), leak=leak, exact=exact(n), content=cont))
    if rank == 0:
        pos = np.sort(omg[omg > -1e-9])
        print(f"n = {n}: omega_h = {np.round(pos, 4).tolist()}  max|gamma| = {np.abs(gam).max():.1e}  leak = {leak:.1e}  "
              f"exact A/F/S = {exact(n)['alfven']:.4f} / {exact(n)['fast']:.4f} / {exact(n)['slow']:.4f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
if rank == 0:
    json.dump(dict(model=a.model, num_elements=ne, L=L, dt=a.dt, beta=BETA, gamma_ad=GAMMA, rows=rows), open(out, "w"))
    print("wrote", out)
