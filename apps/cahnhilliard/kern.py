# Stokes Equations
# ================
#
# A simple example of a saddle-point system, we will use the Stokes
# equations to demonstrate some of the ways we can do field-splitting
# with matrix-free operators.  We set up the problem as a lid-driven
# cavity.
#
# As ever, we import firedrake and define a mesh.::

import os
import sys
import time

# Must be before importing firedrake!
os.environ["OMP_NUM_THREADS"] = str(1)

from firedrake import *

# NOTE: this file is also in exp/apps/firedrake-bench
# TODO: move apps with heavy dependencies out of 'compiler' repo
from firedrake_cahn_hilliard_problem import CahnHilliardProblem

mesh_size = 64
preconditioner = 'fieldsplit'
ksp = 'gmres'
inner_ksp = 'preonly'
max_iterations = 1
degree = 1
dt = 5.0e-06
lmbda = 1.0e-02
theta = 0.5
steps = 1
compute_norms = True
verbose = False

solution_out = None

def solve_ch(res):
    params = CahnHilliardProblem.get_solve_params(
            pc=preconditioner, ksp=ksp, inner_ksp=inner_ksp,
            maxit=max_iterations, verbose=verbose)
    mesh = CahnHilliardProblem.make_mesh(mesh_size)
    u, u0, solver = CahnHilliardProblem.do_setup(mesh, preconditioner,
            degree=degree, dt=dt, theta=theta,
            lmbda=lmbda, params=params)

    file = File(solution_out) if solution_out else None
    CahnHilliardProblem.do_solve(u, u0, solver, steps,
            compute_norms=compute_norms, out_file=file)
