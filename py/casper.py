def invoke_task(loops):
    for l in loops:
        if hasattr(l, "compute"): # some are funcs
            r = l.compute()
        else:
            r = l()
    return r
