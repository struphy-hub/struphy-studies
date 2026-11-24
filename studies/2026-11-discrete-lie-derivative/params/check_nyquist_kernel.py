"""Does the projection of B0 x u lose the shortest axial wave? For smooth velocity 2-forms
u = g(eta1) cos(2 pi (eta2 + n eta3)) e_c (one component c, m = 1, n = 1 .. N3/2) on the kink case, the norm of the
magnetic-field increment b = -curl P(B0 x u), relative to the norm of u (both in the M2 norm), with
- P = Pi^1 (commuting projector, q model: curl T2 u),
- P = L2 projection (variational model with transport_projection="L2": T0 u).

--geom polar (default): IGAPolarCylinder with polar splines (the study's setup);
--geom hollow: HollowCylinder (r = 0.2 .. 1, no axis, no polar splines), same ScrewPinch;
--geom slab: periodic Cuboid (1 x 1 x 2 pi 3) with HomogenSlab (B0 = e_z), no polar splines.

Run: mpirun -n 4 python check_nyquist_kernel.py [--nz 12] [--geom polar]  (table also in nyquist_kernel[_<geom>]_z<nz>.md)
"""

import argparse
import os

import numpy as np
from maybempi import MPI

from cases import domain_and_equil, make_model

ap = argparse.ArgumentParser()
ap.add_argument("--nz", type=int, default=12)
ap.add_argument("--geom", choices=("polar", "hollow", "slab", "torus"), default="polar")
ap.add_argument("--m", type=int, nargs="+", default=[1], help="poloidal wave numbers (default 1); n runs over 0 or 1 .. N3/2")
a = ap.parse_args()
rank = MPI.COMM_WORLD.Get_rank()

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, grids  # noqa: E402

ne = (16, 12, a.nz)
model = make_model("q")
from struphy import domains, equils  # noqa: E402

if a.geom in ("polar", "torus"):
    domain, equil = domain_and_equil("kink" if a.geom == "polar" else "torus", ne)
    bcs, polar = (("free", "dirichlet"), None, None), True
elif a.geom == "hollow":
    domain = domains.HollowCylinder(a1=0.2, a2=1.0, Lz=2 * np.pi * 3.0)
    equil = equils.ScrewPinch(a=1.0, R0=3.0, B0=1.0, q0=0.7, q1=1.3, n1=0.0, n2=0.0, na=1.0, p0=0.1)
    bcs, polar = (("dirichlet", "dirichlet"), None, None), False
else:
    domain = domains.Cuboid(r1=1.0, r2=1.0, r3=2 * np.pi * 3.0)
    equil = equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0, beta=0.1, n0=1.0)
    bcs, polar = (None, None, None), False
sim = Simulation(
    model=model,
    env=EnvironmentOptions(out_folders=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sims"),
                           sim_folder=f"nyq_kernel_{a.geom}"),
    time_opts=Time(dt=0.5, Tend=0.5, split_algo="Strang"),
    domain=domain,
    equil=equil,
    grid=grids.TensorProductGrid(num_elements=ne),
    derham_opts=DerhamOptions(degree=(3, 3, 3), bcs=bcs, polar_splines=polar),
)
sim.allocate()
from struphy.feec.variational_utilities import L2TransportOperator  # noqa: E402
from struphy.propagators.base import Propagator  # noqa: E402

derham, mass_ops = Propagator.derham, Propagator.mass_ops
M2 = mass_ops.M2
T2 = Propagator.basis_ops.T2
curl = derham.curl
T0 = L2TransportOperator(derham, Propagator.domain, mass_ops, assemble=True)
T0.update_coeffs(Propagator.projected_equil.b2)
bop = derham.boundary_ops["2"]


def norm(v):
    return np.sqrt(M2.dot(v).inner(v))


def wave(c, n, m=1):
    u = derham.V2pol.zeros()
    t = u.tp.blocks[c] if polar else u.blocks[c]
    nr = u.space.n_rings[c] if polar else 0
    N = t.space.npts
    s = (np.arange(N[0]) + 0.5) / N[0]
    # slab: no radial profile needed (periodic), keep the same shape for comparability
    full = (np.sin(np.pi * s) ** 2)[:, None, None] * np.cos(
        2 * np.pi * (m * np.arange(N[1])[None, :, None] / N[1] + n * np.arange(N[2])[None, None, :] / N[2])
    )
    full[:nr] = 0.0
    sl = tuple(slice(st, en + 1) for st, en in zip(t.starts, t.ends))
    t[sl] = full[sl]
    u = bop.dot(u)
    u.update_ghost_regions()
    return u


rows = []
for m in a.m:
    for c, name in ((0, "u^1 (radial flux)"), (1, "u^2 (poloidal flux)"), (2, "u^3 (axial flux)")):
        for n in range(0 if a.m != [1] else 1, a.nz // 2 + 1):
            u = wave(c, n, m)
            nu = norm(u)
            bq = curl.dot(T2.dot(u))
            bv = T0.dot(u)
            rows.append((name, m, n, norm(bq) / nu, norm(bv) / nu))
if rank == 0:
    with open(os.path.join(os.environ.get("STUDY_RUNS", os.path.dirname(os.path.abspath(__file__))),
                           f"nyquist_kernel{'' if a.geom == 'polar' else '_' + a.geom}_z{a.nz}"
                           f"{'' if a.m == [1] else '_m' + '-'.join(map(str, a.m))}.md"), "w") as f:
        f.write("| component | m | n | Pi^1 (q) | L2 (var) |\n|---|---|---|---|---|\n")
        f.writelines(f"| {name} | {m} | {n} | {q:.3e} | {v:.3e} |\n" for name, m, n, q, v in rows)
    print(f"{a.geom}, N3 = {a.nz}: |b| / |u| (M2 norms), b = -curl P(B0 x u), u = smooth wave (m, n) in one component")
    print(f"{'component':22s} {'m':>3s} {'n':>3s} {'Pi^1 (q)':>11s} {'L2 (var)':>11s}")
    for name, m, n, q, v in rows:
        print(f"{name:22s} {m:3d} {n:3d} {q:11.3e} {v:11.3e}")
