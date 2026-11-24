"""Condition (K), quantitatively: how much discrete bending does an exact non-bending velocity get?

Trial field: the parallel flow z = alpha(eta3) B0 with alpha = cos(2 pi n eta3), n = 0 .. N3/2 (exactly
L_z B0 = curl(B0 x z) = 0). Its 2-form is alpha * (B0 as a 2-form). It is approximated in V2 (velocity in H(div)) by
- the L2 projection (best approximation in the kinetic norm, uniform density), and
- the commuting projector Pi^2,
and the spurious bending is s_h = |b| / |z_h| (M2 norms), b = curl P(B0,h x z_h) with
- P = L2 (variational model, transport_projection="L2": T0),
- P = Pi^1 with analytic B0 (q model: curl T2).
Reference frequencies at r = 0.5: c_s k (sound), k B_theta (poloidal Alfven), k = 2 pi n / Lz.

--geom polar (default): kink case (IGAPolarCylinder, ScrewPinch, sheared field);
--geom slab: periodic Cuboid with HomogenSlab (B0 = e_z, aligned): s_h should be 0.

Run: mpirun -n 4 python check_kernel_K.py [--nz 12] [--geom polar]
"""

import argparse
import os

import numpy as np
from maybempi import MPI

from cases import domain_and_equil, make_model

ap = argparse.ArgumentParser()
ap.add_argument("--nz", type=int, default=12)
ap.add_argument("--geom", choices=("polar", "slab"), default="polar")
a = ap.parse_args()
rank = MPI.COMM_WORLD.Get_rank()

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids  # noqa: E402

ne = (16, 12, a.nz)
model = make_model("q")
if a.geom == "polar":
    domain, equil = domain_and_equil("kink", ne)
    bcs, polar, Lz = (("free", "dirichlet"), None, None), True, 2 * np.pi * 3.0
else:
    Lz = 2 * np.pi * 3.0
    domain = domains.Cuboid(r1=1.0, r2=1.0, r3=Lz)
    equil = equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0, beta=0.1, n0=1.0)
    bcs, polar = (None, None, None), False
sim = Simulation(
    model=model,
    env=EnvironmentOptions(out_folders=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sims"),
                           sim_folder=f"kernel_K_{a.geom}"),
    time_opts=Time(dt=0.5, Tend=0.5, split_algo="Strang"),
    domain=domain,
    equil=equil,
    grid=grids.TensorProductGrid(num_elements=ne),
    derham_opts=DerhamOptions(degree=(3, 3, 3), bcs=bcs, polar_splines=polar),
)
sim.allocate()
from struphy.feec.mass import L2Projector  # noqa: E402
from struphy.feec.variational_utilities import L2TransportOperator  # noqa: E402
from struphy.linear_algebra.solver import SolverParameters  # noqa: E402
from struphy.propagators.base import Propagator  # noqa: E402

derham, mass_ops = Propagator.derham, Propagator.mass_ops
M2 = mass_ops.M2
T2 = Propagator.basis_ops.T2
curl = derham.curl
T0 = L2TransportOperator(derham, Propagator.domain, mass_ops, assemble=True)
T0.update_coeffs(Propagator.projected_equil.b2)
L2P = L2Projector("Hdiv", mass_ops, solver_params=SolverParameters(tol=1e-13, maxiter=5000))


def norm(v):
    return np.sqrt(M2.dot(v).inner(v))


def parallel_flow(n):
    def comp(i):
        b = (equil.b2_1, equil.b2_2, equil.b2_3)[i]
        return lambda e1, e2, e3: np.cos(2 * np.pi * n * e3) * b(e1, e2, e3)

    funs = [comp(i) for i in range(3)]
    zl2 = L2P(funs, apply_bc=True)
    zpi = derham.P2(funs)
    for z in (zl2, zpi):
        z.update_ghost_regions()
    return zl2, zpi


# reference speeds at r = 0.5 (eta1 = 0.5 for the cylinder; homogeneous for the slab)
e = (np.array([0.5]), np.array([0.0]), np.array([0.0]))
gamma = 5.0 / 3.0
cs = np.sqrt(gamma * float(np.squeeze(equil.p0(*e))) / float(np.squeeze(equil.n0(*e))))
if a.geom == "polar":
    r, R0, q = 0.5, 3.0, 0.7 + (1.3 - 0.7) * 0.5**2
    Bth = r / (q * R0)  # B_theta = B0 r / (q R0), B0 = 1, q(r) = q0 + (q1 - q0) r^2
else:
    Bth = 0.0

rows = []
for n in range(0, a.nz // 2 + 1):
    zl2, zpi = parallel_flow(n)
    k = 2 * np.pi * n / Lz
    rows.append(
        (n, k,
         norm(T0.dot(zl2)) / norm(zl2), norm(curl.dot(T2.dot(zl2))) / norm(zl2),
         norm(T0.dot(zpi)) / norm(zpi), norm(curl.dot(T2.dot(zpi))) / norm(zpi),
         cs * k, Bth * k)
    )
if rank == 0:
    head = (f"{a.geom}, N3 = {a.nz}: spurious bending s_h = |curl P(B0 x z_h)| / |z_h| of the parallel flow "
            f"z = cos(2 pi n eta3) B0\n(c_s = {cs:.3f}, B_theta(r=0.5) = {Bth:.3f})\n")
    tab = ("| n | k | z_h=L2, P=L2 (var) | z_h=L2, P=Pi^1 (q) | z_h=Pi^2, P=L2 | z_h=Pi^2, P=Pi^1 | c_s k | k B_theta |\n"
           "|---|---|---|---|---|---|---|---|\n")
    tab += "".join(f"| {n} | {k:.3f} | {s1:.3e} | {s2:.3e} | {s3:.3e} | {s4:.3e} | {c:.3e} | {t:.3e} |\n"
                   for n, k, s1, s2, s3, s4, c, t in rows)
    out = os.path.join(os.environ.get("STUDY_RUNS", os.path.dirname(os.path.abspath(__file__))),
                       f"kernel_K_{a.geom}_z{a.nz}.md")
    with open(out, "w") as f:
        f.write(head + "\n" + tab)
    print(head + tab)
