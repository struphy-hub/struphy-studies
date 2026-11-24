"""Is the kernel of the field-line bending at the shortest axial wave a property of the velocity space? Periodic slab
(Cuboid 1 x 1 x 2 pi 3, HomogenSlab B0 = e_z, no polar splines, 12 x 12 x 12 elements, degree 3). For Fourier waves of the
spline coefficients u_c = cos(2 pi (k1 i / N1 + k2 j / N2 + k3 l / N3)) in one in-plane component c of the velocity,
the norm of b = -curl P(B0 x u) relative to the norm of u (mass-matrix norms of the spaces), for

- u in H(div) (2-form): P = Pi^1 (curl T2 u) and P = L2 (L2TransportOperator);
- u in H1vec (vector field): P = Pi^1 (curl Tv u) and P = L2 projection of B0 x u onto V1.

Prediction from the spline degrees (E1 = B0 u2 needs degrees (2,3,3), E2 = -B0 u1 needs (3,2,3)):
- H(div), u1 (3,2,2), u2 (2,3,2): mismatch along z -> zero at k3 = N3/2 for any (k1, k2);
- H1vec, u1, u2 (3,3,3): mismatch across B0 -> zero for u2 at k1 = N1/2 and for u1 at k2 = N2/2, for any k3.

Run: mpirun -n 2 python check_velocity_space.py   (table also in velocity_space_kernel.md)
"""

import os

import numpy as np
from maybempi import MPI

from cases import make_model

rank = MPI.COMM_WORLD.Get_rank()

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids  # noqa: E402

NE = (12, 12, 12)
sim = Simulation(
    model=make_model("q"),
    env=EnvironmentOptions(out_folders=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sims"),
                           sim_folder="vel_space_kernel"),
    time_opts=Time(dt=0.5, Tend=0.5, split_algo="Strang"),
    domain=domains.Cuboid(r1=1.0, r2=1.0, r3=2 * np.pi * 3.0),
    equil=equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0, beta=0.1, n0=1.0),
    grid=grids.TensorProductGrid(num_elements=NE),
    derham_opts=DerhamOptions(degree=(3, 3, 3), bcs=(None, None, None), polar_splines=False),
)
sim.allocate()
from struphy.feec.mass import L2Projector  # noqa: E402
from struphy.feec.variational_utilities import L2TransportOperator  # noqa: E402
from struphy.propagators.base import Propagator  # noqa: E402

derham, mass_ops, basis_ops = Propagator.derham, Propagator.mass_ops, Propagator.basis_ops
peq = Propagator.projected_equil
M = {"2": mass_ops.M2, "v": mass_ops.Mv}
curl = derham.curl
T0 = L2TransportOperator(derham, Propagator.domain, mass_ops, assemble=True)
T0.update_coeffs(peq.b2)
P1 = L2Projector("Hcurl", mass_ops)
uf = derham.create_spline_function("u_v", "H1vec")
bf = derham.create_spline_function("b_eq", "Hdiv")
bf.vector = peq.b2


def norm(v, form):
    return np.sqrt(M[form].dot(v).inner(v))


def wave(form, c, k):
    u = (derham.V2pol if form == "2" else derham.Vvpol).zeros()
    t = u.blocks[c]
    N = t.space.npts
    idx = np.meshgrid(*[np.arange(n) for n in N], indexing="ij")
    full = np.cos(2 * np.pi * sum(kk * ii / n for kk, ii, n in zip(k, idx, N)))
    sl = tuple(slice(st, en + 1) for st, en in zip(t.starts, t.ends))
    t[sl] = full[sl]
    u.update_ghost_regions()
    return u


def l2_v(u):
    """-curl P_L2(B0 x u) for u in H1vec: the 1-form components (b2 x u)_mu = eps_{mu a n} b2_a u^n, L2-projected."""
    uf.vector = u

    def comp(mu):
        def f(e1, e2, e3):
            g = (e1[:, 0, 0], e2[0, :, 0], e3[0, 0, :])
            uu, bb = uf(*g, local=True), bf(*g, local=True)  # points of this rank only
            a, n = (mu + 1) % 3, (mu + 2) % 3
            return bb[a] * uu[n] - bb[n] * uu[a]

        return f

    E = P1.solve(P1.get_dofs([comp(0), comp(1), comp(2)]))
    return curl.dot(E)


ops = {
    ("2", "Pi^1"): lambda u: curl.dot(basis_ops.T2.dot(u)),
    ("2", "L2"): lambda u: T0.dot(u),
    ("v", "Pi^1"): lambda u: curl.dot(basis_ops.Tv.dot(u)),
    ("v", "L2"): l2_v,
}
K3 = (1, 3, 5, 6)
PERP = [(0, 0), (3, 0), (6, 0), (0, 3), (0, 6)]
rows = []
for form, sname in (("2", "H(div)"), ("v", "H1vec")):
    for c, cname in ((0, "u1 (x)"), (1, "u2 (y)")):
        for k1, k2 in PERP:
            vals = {}
            for proj in ("Pi^1", "L2"):
                for k3 in K3:
                    u = wave(form, c, (k1, k2, k3))
                    vals[proj, k3] = norm(ops[form, proj](u), "2") / norm(u, form)
            rows.append((sname, cname, k1, k2, vals))

if rank == 0:
    head = "| velocity | component | k1 | k2 | " + " | ".join(f"k3 = {k}: Pi^1 / L2" for k in K3) + " |"
    lines = [head, "|" + "---|" * (4 + len(K3))]
    for sname, cname, k1, k2, v in rows:
        lines.append(f"| {sname} | {cname} | {k1} | {k2} | "
                     + " | ".join(f"{v['Pi^1', k]:.2e} / {v['L2', k]:.2e}" for k in K3) + " |")
    out = os.path.join(os.environ.get("STUDY_RUNS", os.path.dirname(os.path.abspath(__file__))),
                       "velocity_space_kernel.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
