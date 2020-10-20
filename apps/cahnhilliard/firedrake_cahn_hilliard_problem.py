from firedrake import *

from mpi4py import MPI
comm = MPI.COMM_WORLD

class CahnHilliardProblem:

    def make_mesh(x, dim=2):
        return UnitSquareMesh(x, x) if dim == 2 else UnitCubeMesh(x, x, x)

    def get_solve_params(verbose=False, maxit=1, ksp='gmres',
            inner_ksp='preonly', pc='fieldsplit'):
        params = {'pc_type': pc,
                  'ksp_type': ksp,
                  #'ksp_monitor': True,
                  'snes_monitor': None,

                  #'snes_rtol': 1e-9,
                  #'snes_atol': 1e-10,
                  #'snes_stol': 1e-14,

                  'snes_rtol': 1e-5,
                  'snes_atol': 1e-6,
                  'snes_stol': 1e-7,

                  #'snes_rtol': 1e-4,
                  #'snes_atol': 1e-6,
                  #'snes_stol': 1e-8,

                  #'snes_rtol': 1e-2,
                  #'snes_atol': 1e-4,
                  #'snes_stol': 1e-6,
                  'snes_linesearch_type': 'basic',
                  'snes_linesearch_max_it': 1,
                  #'ksp_rtol': 1e-6,
                  #'ksp_atol': 1e-15,

                  #'ksp_rtol': 1e-5,
                  #'ksp_atol': 1e-9,

                  'ksp_rtol': 1e-4,
                  'ksp_atol': 1e-8,
                  'pc_fieldsplit_type': 'schur',
                  'pc_fieldsplit_schur_factorization_type': 'lower',
                  'pc_fieldsplit_schur_precondition': 'user',
                  'fieldsplit_0_ksp_type': inner_ksp,
                  'fieldsplit_0_ksp_max_it': maxit,
                  'fieldsplit_0_pc_type': 'hypre',
                  'fieldsplit_1_ksp_type': inner_ksp,
                  'fieldsplit_1_ksp_max_it': maxit,
                  'fieldsplit_1_pc_type': 'mat'}
        if verbose:
            params['ksp_monitor'] = True
            params['snes_view'] = True
            params['snes_monitor'] = True
        return params

    def do_setup(mesh, pc, degree=1, theta=0.5, dt=5.0e-06,
                lmbda=1.0e-02, inner_ksp='preonly', maxit=1, params={}):
        V = FunctionSpace(mesh, "Lagrange", degree)
        ME = V*V

        # Define trial and test functions
        du = TrialFunction(ME)
        q, v = TestFunctions(ME)

        # Define functions
        u = Function(ME)   # current solution
        u0 = Function(ME)  # solution from previous converged step

        # Split mixed functions
        dc, dmu = split(du)
        c, mu = split(u)
        c0, mu0 = split(u0)

        # Create intial conditions and interpolate
        init_code = "A[0] = 0.63 + 0.02*(0.5 - (double)random()/RAND_MAX);"
        user_code = """int __rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &__rank);
        srandom(2 + __rank);"""
        par_loop(init_code, direct, {'A': (u[0], WRITE)},
                 headers=["#include <stdlib.h>"], user_code=user_code)
        u.dat.data_ro

        # Compute the chemical potential df/dc
        c = variable(c)
        f = 100*c**2*(1-c)**2
        dfdc = diff(f, c)

        mu_mid = (1.0-theta)*mu0 + theta*mu

        # Weak statement of the equations
        F0 = c*q*dx - c0*q*dx + dt*dot(grad(mu_mid), grad(q))*dx
        F1 = mu*v*dx - dfdc*v*dx - lmbda*dot(grad(c), grad(v))*dx
        F = F0 + F1

        # Compute directional derivative about u in the direction of du (Jacobian)
        J = derivative(F, u, du)

        problem = NonlinearVariationalProblem(F, u, J=J)
        solver = NonlinearVariationalSolver(problem, solver_parameters=params)

        if pc in ['fieldsplit', 'ilu']:
            sigma = 100
            # PC for the Schur complement solve
            trial = TrialFunction(V)
            test = TestFunction(V)
            mass = assemble(inner(trial, test)*dx).M.handle
            a = 1
            c = (dt * lmbda)/(1+dt * sigma)
            hats = assemble(sqrt(a) * inner(trial, test)*dx + sqrt(c)*inner(grad(trial), grad(test))*dx).M.handle

            from firedrake.petsc import PETSc
            ksp_hats = PETSc.KSP()
            ksp_hats.create()
            ksp_hats.setOperators(hats)
            opts = PETSc.Options()

            opts['ksp_type'] = inner_ksp
            opts['ksp_max_it'] = maxit
            opts['pc_type'] = 'hypre'
            ksp_hats.setFromOptions()

            class SchurInv(object):
                def mult(self, mat, x, y):
                    tmp1 = y.duplicate()
                    tmp2 = y.duplicate()
                    ksp_hats.solve(x, tmp1)
                    mass.mult(tmp1, tmp2)
                    ksp_hats.solve(tmp2, y)

            pc_schur = PETSc.Mat()
            pc_schur.createPython(mass.getSizes(), SchurInv())
            pc_schur.setUp()
            pc = solver.snes.ksp.pc
            pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER, pc_schur)

        return u, u0, solver

    def do_measure_overhead(u0, solver):
        for _ in range(100):
            u0.assign(u)
            solver.solve()

    def do_solve(u, u0, solver, steps, compute_norms=False, out_file=None):
        for step in range(steps):
            u0.assign(u)
            solver.solve()
            if out_file is not None:
                out_file << (u.split()[0], step)
            if compute_norms:
                nu = norm(u)
                if comm.rank == 0:
                    print(step, 'L2(u):', nu)
