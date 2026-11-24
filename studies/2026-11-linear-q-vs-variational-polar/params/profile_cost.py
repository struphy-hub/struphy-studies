"""Cost per time step of a linear model: wall time per propagator and the iterations of the outer and inner (nested)
linear solves, averaged over N steps after one warm-up step, polar cylinder 16 x 12 x 12.

Run (4 MPI ranks): mpirun -n 4 python profile_cost.py --model var --case kink --dt 0.125 --steps 8 --out cost_var_dt0.125.json
"""

import argparse
import collections
import json
import os
import time

import feectools.linalg.solvers as solvers
from maybempi import MPI

from cases import domain_and_equil, make_model
from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, grids

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["q", "var", "var_pi"], required=True)
ap.add_argument("--case", default="kink")
ap.add_argument("--dt", type=float, required=True)
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rank = MPI.COMM_WORLD.Get_rank()

current, depth = [None], [0]
S = collections.defaultdict(lambda: [0, 0, 0.0])  # (propagator, solver, outer/inner) -> solves, iterations, time
for nm in dir(solvers):
    cls = getattr(solvers, nm)
    if isinstance(cls, type) and cls.__module__ == solvers.__name__ and hasattr(cls, "solve"):

        def wrap(orig, cname=nm):
            def solve(self, b, out=None):
                depth[0] += 1
                t0 = time.perf_counter()
                try:
                    return orig(self, b, out=out)
                finally:
                    depth[0] -= 1
                    s = S[(current[0], cname, "inner" if depth[0] > 0 else "outer")]
                    s[0] += 1
                    s[1] += int((getattr(self, "_info", None) or {}).get("niter", 0) or 0)
                    s[2] += time.perf_counter() - t0

            return solve

        cls.solve = wrap(cls.solve)

ne = (16, 12, 12)
model = make_model(a.model)
domain, equil = domain_and_equil(a.case, ne)
sim = Simulation(
    model=model,
    env=EnvironmentOptions(out_folders=os.path.join(os.path.dirname(os.path.abspath(a.out)), "sims"), sim_folder="cost"),
    time_opts=Time(dt=a.dt, Tend=a.dt, split_algo="Strang"),
    domain=domain,
    equil=equil,
    grid=grids.TensorProductGrid(num_elements=ne),
    derham_opts=DerhamOptions(degree=(3, 3, 3), bcs=(("free", "dirichlet"), None, None), polar_splines=True),
)
sim.allocate()
props = vars(model.propagators)
T = collections.defaultdict(float)
for pn, pr in props.items():
    cls = type(pr)
    if getattr(cls, "_cost_wrapped", False):
        continue
    orig = cls.__call__

    def make(orig):
        def call(self, dt):
            name = [k for k, v in props.items() if v is self][0]
            current[0] = name
            t0 = time.perf_counter()
            orig(self, dt)
            T[name] += time.perf_counter() - t0

        return call

    cls.__call__ = make(orig)
    cls._cost_wrapped = True
# a small smooth perturbation so that the solves are not trivial
from struphy.propagators.base import Propagator  # noqa: E402

v = model.mhd.velocity.spline.vector
for blk in v.tp.blocks:
    blk._data[:] = 1e-6
Propagator.derham.boundary_ops["2"].dot(v).copy(out=v)
v.update_ghost_regions()
model.integrate(a.dt, "Strang")  # warm-up: preconditioners for this dt
T.clear()
S.clear()
t0 = time.perf_counter()
for _ in range(a.steps):
    model.integrate(a.dt, "Strang")
tot = (time.perf_counter() - t0) / a.steps
res = dict(model=a.model, case=a.case, dt=a.dt, steps=a.steps, s_per_step=tot, s_per_time=tot / a.dt,
           propagators={k: T[k] / a.steps for k in props},
           solves=[dict(prop=k[0], solver=k[1], kind=k[2], solves_per_step=v[0] / a.steps, its_per_solve=v[1] / max(v[0], 1),
                        s_per_step=v[2] / a.steps) for k, v in S.items()])
if rank == 0:
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"COST {a.model} dt={a.dt}: {tot:.3f} s/step, {tot / a.dt:.2f} s per unit time", flush=True)
    for k, t in res["propagators"].items():
        print(f"  {k:22s} {t:.3f} s/step", flush=True)
