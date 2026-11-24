"""Time runs of the study "linear q model vs. variational p model on polar splines" (energy behaviour): one of the two
linear models on the polar cylinder of a case of ``cases.py``, started from a smooth helical velocity perturbation
(m = 1, n = 1, amplitude 1e-6, divergence free in the plane), with the energy scalars saved in every step.

- ``--model q``: LinearMHDWithQ (B in H(div)). Its scalar ``en_tot`` is the quadratic energy of the perturbation; the
  linear equations do not conserve it when the equilibrium carries a current or a pressure gradient (it exchanges
  energy with the background), and the scheme has no other discrete invariant.
- ``--model var`` / ``var_pi``: VariationalLinearMHD (p, L2 / commuting transport). Its ``en_tot`` contains the
  second-order fields (b-hat, p-hat) and is conserved exactly by every step.

Run: mpirun -n 4 python params_linear.py --model var --case stable --dt 0.5 --tend 100 --sim-folder var_stable_dt0.5
"""

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument("--out-folder", default=str(Path(__file__).resolve().parent), help="Folder of the output folder.")
parser.add_argument("--sim-folder", default="sim_data", help="Name of the output folder.")
parser.add_argument("--model", choices=["q", "var", "var_pi"], default="var")
parser.add_argument("--case", default="stable", choices=["kink", "stable", "suydam"])
parser.add_argument("--dt", type=float, default=0.5)
parser.add_argument("--tend", type=float, default=100.0)
parser.add_argument("--num-elements", type=int, nargs=3, default=[16, 12, 12])
parser.add_argument("--save-velocity", action="store_true", help="also save the velocity (every step).")
parser.add_argument("--test-run", action="store_true", help="8 x 6 x 4 elements, two time steps.")
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements, args.tend = [8, 6, 4], 2 * args.dt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from cases import CASES, domain_and_equil, make_model  # noqa: E402
from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, grids  # noqa: E402
from struphy.initial.base import Perturbation  # noqa: E402

name = f"Linear MHD on a polar cylinder ({args.model}, {args.case})"
description = __doc__

Lz = 2 * np.pi * CASES[args.case]["R0"]


class HelicalSeed(Perturbation):
    """Cartesian velocity component ``comp`` of amp (1 - r^2) (cos(2 pi z/Lz), sin(2 pi z/Lz), 0). Self-contained
    (Struphy stores its source and its ``params`` in the run metadata and rebuilds it from them)."""

    def __init__(self, comp: int, Lz: float, amp: float = 1e-6):
        self.params = dict(comp=comp, Lz=Lz, amp=amp)
        self.given_in_basis = "physical"
        self.comp = comp
        self.Lz = Lz
        self.amp = amp

    def __call__(self, x, y, z, flat_eval=False):
        import numpy as np

        env = self.amp * (1.0 - x**2 - y**2)
        if self.comp == 0:
            return env * np.cos(2 * np.pi * z / self.Lz)
        if self.comp == 1:
            return env * np.sin(2 * np.pi * z / self.Lz)
        return 0.0 * x


model = make_model(args.model)
model.mhd.velocity.save_data = args.save_velocity
model.em_fields.b_field.save_data = False
for comp in (0, 1):
    model.mhd.velocity.add_perturbation(perturbation=HelicalSeed(comp, Lz))

domain, equil = domain_and_equil(args.case, tuple(args.num_elements))
sim = Simulation(
    model=model,
    name=name,
    description=description,
    params_path=__file__,
    env=EnvironmentOptions(out_folders=args.out_folder, save_step=1, sim_folder=args.sim_folder),
    time_opts=Time(dt=args.dt, Tend=args.tend, split_algo="Strang"),
    domain=domain,
    equil=equil,
    grid=grids.TensorProductGrid(num_elements=tuple(args.num_elements)),
    derham_opts=DerhamOptions(degree=(3, 3, 3), bcs=(("free", "dirichlet"), None, None), polar_splines=True),
)

if __name__ == "__main__":
    sim.run()
