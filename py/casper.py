import firedrake

def invoke_task(loops):
    for l in loops:
        if hasattr(l, "compute"): # some are funcs
            r = l.compute()
        else:
            r = l()
    return r

def invoke_task_by_name(ctx, task, state):
    print("invoke_task_by_name: task", task)
    r = invoke_task(ctx[0][task])
    if isinstance(r, firedrake.matrix.Matrix):
        state[task] = r.M.handle

def assemble(f):
    return firedrake.assemble(f, collect_loops=True, allocate_only=False)

def assign(dest, src):
    return dest.assign(src, compute=False)

