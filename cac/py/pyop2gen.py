import importlib

def codegen(gen_module, gen_func):
    gen_mod = importlib.import_module(gen_module)
    print(f"codegen {gen_module}.{gen_func}")
    loops = gen_mod.generate()
    for task, loops in loops.items():
        print(f"task {task}: {len(loops)} loops")
        for loop in loops:
            if hasattr(loop, "compute"): # some are funcs
                loop._jitmodule
