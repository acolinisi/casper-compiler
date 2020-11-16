import os
import importlib

# output dir for compiled kernel libraries: working dir of metaprog
# must be set before importing (see firedrake_configuration), and
# must use env vars, because firedrake resets config when re-imported
out_dir = "fd_kernels"
os.environ["PYOP2_CACHE_DIR"] = os.path.join(out_dir, "pyop2")
os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] = os.path.join(out_dir, "tsfc")

from firedrake import *

def codegen(gen_module, gen_func):
    gen_mod = importlib.import_module(gen_module)
    print(f"codegen {gen_module}.{gen_func}")
    explicit_loops, solver = gen_mod.generate()
    implicit_loops = dict(
            _jac=solver._ctx._assemble_jac,
            _residual=solver._ctx._assemble_residual
            )
    if solver._ctx.Jp is not None:
        implicit_loops["_pjac"]=solver._ctx._assemble_pjac

    for loop_set in [explicit_loops, implicit_loops]:
        for task, loops in loop_set.items():
            print(f"task {task}: {len(loops)} loops")
            for loop in loops:
                if hasattr(loop, "compute"): # some are funcs
                    loop._jitmodule
