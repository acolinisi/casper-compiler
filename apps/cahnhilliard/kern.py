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
compute_norms = False # TODO
verbose = False

solution_out = None

def invoke_loops(loops):
    for l in loops:
        if hasattr(l, "compute"): # some are funcs
            r = l.compute()
        else:
            r = l()
    return r


def generate():
    # TODO: should not need to actually create the mesh
    mesh = CahnHilliardProblem.make_mesh(mesh_size)
    init_loop, mass_loops, hats_loops, assign_loops, u, u0, solver = \
            CahnHilliardProblem.do_setup(mesh, pc=preconditioner,
            degree=degree, dt=dt, theta=theta,
            lmbda=lmbda, ksp=ksp, inner_ksp=inner_ksp,
            maxit=max_iterations, verbose=verbose,
            # TODO: shouldn't be exposed to the developer
            out_lib_dir=os.path.join(os.getcwd(), 'fd_kernels'))

# TODO: can the state object be any user-defined class instead of dict?
def setup(state):
    print("init_ch")

    mesh = CahnHilliardProblem.make_mesh(mesh_size)
    # TODO: not everything in do_setup needs to be done
    init_loop, mass_loops, hats_loops, assign_loops, u, u0, solver = \
            CahnHilliardProblem.do_setup(mesh, pc=preconditioner,
            degree=degree, dt=dt, theta=theta,
            lmbda=lmbda, ksp=ksp, inner_ksp=inner_ksp,
            maxit=max_iterations, verbose=verbose)

    state["init_loop"] = init_loop
    state["mass_loops"] = mass_loops
    state["hats_loops"] = hats_loops
    state["assign_loops"] = assign_loops

    state["u"] = u
    state["u0"] = u0
    state["solver"] = solver

def init(state):
    invoke_loops([state["init_loop"]])

def assemble_mass(state):
    mass_m = invoke_loops(state["mass_loops"])
    state["mass"] = mass_m.M.handle

def assemble_hats(state):
    hats_m = invoke_loops(state["hats_loops"])
    state["hats"] = hats_m.M.handle

def solve(state):
    file = File(solution_out) if solution_out else None
    CahnHilliardProblem.do_solve(state["mass"], state["hats"],
            state["assign_loops"],
            state["u"], state["u0"], state["solver"], steps,
            maxit=max_iterations, inner_ksp=inner_ksp,
            compute_norms=compute_norms, out_file=file)
