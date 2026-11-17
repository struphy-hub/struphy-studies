"""Shared setup of the study "linear q model vs. variational p model on polar splines": the two models with their
options, the equilibria of the test cases and the polar domains.

Models
------
- ``q``: :class:`~struphy.models.LinearMHDWithQ` with the magnetic field in H(div) (non-variational, pressure via
  q = sqrt(p)).
- ``var``: :class:`~struphy.models.VariationalLinearMHD` with the velocity in H(div), pressure p
  (``model="linear_p"``), the L2 projections in the transport (``transport_projection="L2"``, struphy#950) and the
  Kronecker preconditioner of the Schur complement of the sound step (``schur_precond``, struphy#951).
- ``var_pi``: the same with the commuting projectors in the transport (``transport_projection="commuting"``), the
  scheme before struphy#950 (spurious mode at the axis).

Cases (polar cylinder, screw pinch, uniform density, B0 = 1, a = 1, length 2 pi R0; or polar torus)
-----------------------------------------------------------------------------------------------------
- ``kink``: q = 0.7 -> 1.3, R0 = 3: m = 1 internal kink (q = 1 at r = 0.71), block (m, n) = (1, -1).
- ``stable``: q = 1.1 -> 1.9, R0 = 3: no q = 1 surface, block (1, -1) without resonance: stable, shear Alfven
  continuum omega_A(r) = |m/q(r) + n| B0 / R0 and discrete modes.
- ``suydam``: q = 1.5 -> 2.5, R0 = 2: pressure-driven (Suydam-unstable) at q = 2 (r = 0.71), block (2, -1).
- ``torus``: IGAPolarTorus (a = 1, R0 = 3) with AdhocTorus (q = 0.8 -> 2.0, beta = 0.1; NOT an exact equilibrium),
  toroidal block n = -1.
"""

import numpy as np

from struphy import domains, equils
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters

CASES = {
    "kink": dict(geom="cyl", q0=0.7, q1=1.3, R0=3.0, p0=0.1, mn=(1, -1)),
    "stable": dict(geom="cyl", q0=1.1, q1=1.9, R0=3.0, p0=0.1, mn=(1, -1)),
    "suydam": dict(geom="cyl", q0=1.5, q1=2.5, R0=2.0, p0=0.1, mn=(2, -1)),
    "torus": dict(geom="torus", q0=0.8, q1=2.0, R0=3.0, beta=0.1, mn=(1, -1)),
}


def domain_and_equil(case: str, num_elements: tuple):
    """The polar domain and the equilibrium of ``case`` (``num_elements`` = (radial, poloidal, axial))."""
    c = CASES[case]
    if c["geom"] == "torus":
        domain = domains.IGAPolarTorus(
            num_elements=tuple(num_elements[:2]), degree=(3, 3), a=1.0, R0=c["R0"], tor_period=1
        )
        equil = equils.AdhocTorus(
            a=1.0, R0=c["R0"], B0=1.0, q_kind=0, q0=c["q0"], q1=c["q1"], n1=0.0, n2=0.0, na=1.0, p_kind=1,
            beta=c["beta"],
        )
    else:
        domain = domains.IGAPolarCylinder(
            num_elements=tuple(num_elements[:2]), degree=(3, 3), a=1.0, Lz=2 * np.pi * c["R0"]
        )
        equil = equils.ScrewPinch(
            a=1.0, R0=c["R0"], B0=1.0, q0=c["q0"], q1=c["q1"], n1=0.0, n2=0.0, na=1.0, p0=c["p0"]
        )
    return domain, equil


def make_model(name: str, tol: float = 1e-12):
    """The model ``name`` ("q", "var", "var_pi") with the options of the study (solver tolerance ``tol``)."""
    sp = SolverParameters(tol=tol, maxiter=5000)
    if name == "q":
        import dataclasses

        from struphy.models import LinearMHDWithQ

        model = LinearMHDWithQ(b_space="Hdiv")
        for prop in vars(model.propagators).values():
            opts = prop.Options()
            kw = {f: sp for f in ("solver_params", "solver_params_M3") if hasattr(opts, f)}
            if kw:
                prop.options = dataclasses.replace(opts, **kw)
        return model
    from struphy.models import VariationalLinearMHD

    tp = "L2" if name == "var" else "commuting"
    nl = NonlinearSolverParameters(tol=tol, maxiter=100)
    model = VariationalLinearMHD(with_viscosity=False, with_resistivity=False, u_space="Hdiv")
    pr = model.propagators
    pr.variat_dens.options = pr.variat_dens.Options(model="linear", solver_params=sp, nonlin_solver=nl)
    pr.variat_lin_alfven.options = pr.variat_lin_alfven.Options(
        schur_precond="StiffnessPreconditioner", solver_params=sp, transport_projection=tp
    )
    pr.variat_lin_sound.options = pr.variat_lin_sound.Options(
        model="linear_p", solver_params=sp, nonlin_solver=nl, transport_projection=tp,
        schur_precond="StiffnessPreconditioner",
    )
    return model


def model_vectors(model, name: str):
    """(velocity, magnetic field, thermodynamic variable, density) coefficient vectors of the model."""
    thermo = model.mhd.sqrt_p if name == "q" else model.mhd.pressure
    return (
        model.mhd.velocity.spline.vector,
        model.em_fields.b_field.spline.vector,
        thermo.spline.vector,
        model.mhd.density.spline.vector,
    )
