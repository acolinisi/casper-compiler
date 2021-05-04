CASPER Compiler
===============

Overview
--------

The CASPER Compiler allows writing applications that are composed of tasks,
where each task may be written in a DSL. The current list of supported DSLs is:
* Halide: a declarative language for expressing computation over regular arrays
  without explicit iteration or control flow,
* Unified Form Language (UFL): a declarative language for defining assembled
  matrices for Finite Element Methods in terms of differential equation terms,
* Non-DSL: tasks written in a general-purpose language C, C++, or Python.

The CASPER Compiler provides a mechanism for joining tasks written in the DSLs
into one graph that represents the computation in a complete program. CASPER
invokes the respective DSL compiler to compile each task into object code: the
Halide compiler for Halide tasks, and The Structure-Preserving Form Compiler
(TSFC) for UFL. CASPER Compiler then links the object code produced by these
DSL compilers into one binary.

The notion of "compilation" in the context of CASPER and DSLs embedded into
a host language differs from the familiar parse-lex-compile process for a
standalone (non-embedded) programming language. In CASPER, the application
binary is produced by a meta-program binary that is compiled by the C++ host
compiler and linked against the CASPER Compiler library. It's called a
meta-program because it is a program, that when executed, produces another
program (the application). The CASPER Compiler does not parse DSL code of the
tasks nor the meta-program code. The CASPER Compiler is the library that is
linked into the meta-program and provides the C++ API for defining the tasks
and the data buffers, and linking them together into a graph. The DSLs that
CASPER supports are embedded DSLs (into C++ and into Python, respectively), and
thus the source code for a task is itself a part of the meta-program, i.e. when
the code that defines the task executes, it produces object code for the task,
not the output of the numerical algorithm in the task. In summary, one might
say that for each application, a separate instance of the CASPER Compiler
is first defined and built, and only then executed to produce the application
binary. Some example meta-programs are presented in a later section of this
document.

Depending on the DSL which a task is written in, the task code is in
a separate source file specific to that DSL. For example, Halide tasks
are defined in a `.cpp` file by a generator class written in the C++ (host
language) with statements that are Halide directives. Tasks written in
UFL are in `.py` files with a generator method that returns the objects
constructed by Python statements that are UFL directives.

CASPER Compiler features include:
* Compilation of the DSL code into multiple variants of object code
  and inclusion of all variants into the binary for selection at runtime.
* Automatic instrumentation of the code that profiles the execution time
  of each task.
* [*WIP*] Automatic invocation of the CASPER Autotuner to select the
  best performing variants for inclusion into the application binary.

Implementation
--------------

The components of the CASPER Compiler implement the following responsibilities:
* the front-end constructs the task graph (CASPER's intermediate
  representation),
* the back-end accepts the (declarative) in-meory task graph and
  constructs a (procedural) skeleton program from it, and then lowers that
  program into object code
* the back-end also invokes the respective DSL compiler for each task
  that emits the object code for each task
* a CMake build code links the skeleton program and the tasks into an
  executable.

Front-end
---------

The class hierarchy that is defined by the CASPER meta-program API includes
classes for objects corresponding to different types of tasks and for datasets.
The task objects are parametrized by which DSL they are written in. The dataset
objects are parametrized by their dimensions and element type (float, double,
integer, etc).

Each task is written using a generator. This is characteristic of an
embedded DSL. The generator is part of the meta-program, and when the
meta-program runs it executes the generator, which in turn outputs
the object code. Halide tasks are written using the `Halide::Generator`
class shipped by Halide, and Python tasks are written using a `generate()`
method at the file scope, described later in this document.

Dataset objects are created first and then passed to task object constructors.
When two tasks operate on the same dataset (in sequence), then the same dataset
object is passed to constructors of both tasks. Task objects are linked into a
graph by passing previously created task objects into constructors for the task
objects that follow. The back-end will use the edges in this task graph to
determine the order in which to invoke the tasks.

Note that only Directed-Acyclic Graphs (DAGs) of tasks are supported. This
means that there cannot be a backedge from a task to one of its dependencies.
Iteration cannot be expressed in this task-based model, the back-end cannot
emit loops: each task will be invoked exactly once. All iteration is contained
within individual tasks, i.e. within the code that implements the task.

Back-end
--------

The input to the back-end is the in-memory data structure that represents
the task graph, that had been constructed by the front-end. The back-end
lowers this task graph representation into native object code, through
several intermediary representations.

In the lowering process, the second intermediate representation (after the task
graph) is a Multi-Level IR (MLIR) representation. MLIR is part of the LLVM
project; it higher level representation than LLVM IR and can be lowered into
LLVM IR. Rather than having a fixed set of operators that roughly correspond to
native instructions, as does LLVM IR, the operations in MLIR are all
custom-defined operators. MLIR ships with some sets of operations, called
"dialects," and CASPER Compiler defines an additional dialect. The MLIR program
may have operators from different dialects at the same time. Lowering is a
process of rewriting operators in one dialect in terms of operators in other
dialects, thereby shrinking the set of dialects in use, until the only dialect
remaining is the LLVM IR, which can be lowered into native object code.

The lowering process (defined in `cac/mlir.pp`) in CASPER reduces the set of
dialects in this order:

    [Casper Task Graph]
    {CasperDialect, StandardOpsDialect}
    {StandardOpsDialect, AffineDialect, SCFDialect, LLVMDialect}
    {LLVMDialect}
    [native object code]

Note: The legacy name for `CasperDialect` carried over from development is
`ToyDialect`.

The code in `cac/build.cpp` lowers the task graph into an MLIR program. The
essence of that lowering is to emit an abstract CASPER-defined task-invocation
operator for each task. Different operators are different for different DSLs,
and each instance is parametrized by task-specific information, such as it's
entry function, and the list of datasets that the task accesses.

The lowering from `{CasperDialect, StandardOpsDialect}` into a set of less
abstract dialects shipped with MLIR distribution (`{StandardOpsDialect,
AffineDialect, SCFDialect, LLVMDialect}`) is implemented in
`cac/mlir/LowerToAffineLoops.cpp`. The task invocation operators are lowered
into Call operator, each of which will corespond to a call into the entry point
of the respective task.  That entry point is carried as a parameter of the
operator.

The CASPER dataset objects are lowered into Tensor objects of the StandardOps
dialect. A Tensor type is an abstract representation of multi-dimensional
arrays that is later lowered into a MemRef type, that represents a concrete
region of memory, with support for different memory layouts. When a MemRef object
is to be passed into a task's entry function, CASPER Compiler converts the
MemRef datatype into a data type native to the the task's DSL. For example,
for Halide, MemRef is converted into a `halide_buffer_t` structure. This
structure is created on the stack and populated from the MemRef fields,
with the central field being the pointer to data. No data is every copied,
only a descriptor that points to the data memory is being changed. This
conversion is implemented in `cac/mlir/LowerToLLVM.cpp`.

### Compiling the tasks

Besides lowering the task graph that represents the whole program, it is also
necessary to compile the DSL code for each task into native object code. CASPER
Compiler does this before processing the task graph, by invoking the respective
DSL compilers.

The meta-program invokes each DSL compiler. This kind of programatic invocation
of a compiler is characteristic for languages embedded into another language:
building and running the meta-program runs the DSL compiler -- there's not a
separate OS process that would take files as input and parse them, as
would be the case for a non-embedded language. In other terminology, the
DSL code is written in the context of a generator, the generator is compiled
and linked into the meta-program, and the meta-program invokes the generator
when the meta-program executes.

#### Tasks written in Halide

Tasks that elect to use the Halide language are written in a `.cpp` file, and
use the `Halide::Generator` class. The DSL statements are written in the
`generate()` method, and the computation organization ("schedule") is written
in the `schedule()` method. The fields of type `GeneratorParam` may be set to
create different variants of the object code, as described below. The name of
the entrypoint (or, the prefix of the name in case of multiple variants -- see
below) is given to the `HALIDE_REGISTER_GENERATOR` macro. The CASPER
meta-program invokes some top-level routines in the Halide compiler library for
each generator, and Halide eventually invokes the generate and the schedule
methods to produce a library archive (`.a`) with the object code.

Halide tasks may input and output into `Buffer<float>`, `Buffer<double>`,
`Buffer<int>` multidimentional array types, as well as into scalar variables
of type `int` or `double`. The lowering pass in the CASPER Compiler described
previously, converts MemRef types into these Halide types, as well as wrapps
scalars into a "box" so that they may be dereferenced and modified by the
Halide task function.

CASPER supports more than one variant of object code compiled from the same
source code of one task. For example, a Halide task might have parameters in
its Halide schedule, and different choices for the values of these parameters
would correspond to different variants of the compiled object code. The
variants are defined in a configuration file: `tuned-params.ini` that is
expected to be in application's directory, described in the usage section.
When the meta-program executes, it invokes the DSL compiler once for each
variant defined in this file. The object code for all variants is eventually
linked into the same executable, and this is possible because the entry function
into each variant is given a unique suffix. The CASPER Autotuner is useful
for automatically finding parameter values with the best performance.

#### Tasks written in Unified Form Language (UFL)

Tasks that elect to use UFL to express differential equations
are written in `.py` file with two special methods at the top-level scope:
* a `generate()` method that returns a list of kernels to be compiled (details
  below) that are defined by UFL expressions
* a `solve()` method that is invoked only at runtime

CASPER Compiler leverages UFL through the Firedrake framework, however with one
important change: compilation in CASPER is ahead-of-time, whereas compilation
in Firedrake is just-in-time. This change introduces a complication to the
architecture of the UFL tasks. In Firedrake the information needed to compile
is tightly coupled with the information/state needed to run the program. CASPER
Compiler does patch Firedrake to decouple these two sets of state but not to a
perfect extent. As a result, the UFL expressions and other objects instantiated
in the `generate()` method are necessary not only to compile but also to run
the program. Therefore, CASPER Compiler invokes the `generate` method both at
compile time and again at rutime, however during the runtime invocation, no
compilation happens, only various runtime state objects are setup.

For the above reaons, the `generate()` method returns a tuple, the elements of
which are:
* a list of objects each of which corresponds to a kernel that will later be
  compiled,
* a solver object that defines and parametrizes the solver routines
  from `petsc4py` that are to be used for the solution of equations;
  this is part of the runtime state
* a dictionary of other objects that are part of the runtime state

The UFL expressions are written in the `generate()` method, and the return
value from each expression is accumulated into a list. That list is
the first element of the tuple returned from the `generate` method.

The `solve()` method takes a context object and a state object. The
context object contains the solver object and the dictionary returned by the
`generate()` method in the second and third elements of its return tuple. The
state object contains references to the compiled kernels. The code in the
`solve()` tasks uses casper API method `invoke_task()` to invoke such a
compiled kernel.  The `solve()` method may also contain any arbitrary Python
code, and will certainly contain calls to the `solver.solve()` method
and calls into the PETSc library via `petsc4py`.

### Construction of the executable

The output from the lowering transformations is a program in LLVM IR,
saved in a standalone `.ll` file. The final stage of the CASPER Compiler
is to compile this LLVM IR program into a native binary. CASPER Compiler
does this by calling `llc`, the LLVM compiler. This process is orchestrated
by CMake build code that is part of the CASPER Compiler.

Applications that are built by CASPER, include the CASPER CMake package
(defined in `CASPERConfig.cmake` shipped with CASPER Compiler). This 
CMake package defines rules for:
1. compiling and linking the meta-program
2. executing the meta-program to produce the `.ll` LLVM IR code for the
   task graph (lowering described above takes place here), and the
   object code archives for each variant of each task
3. compiling the `.ll` LLVM IR code into an object code archive (`.a`)
   with the lowered task-graph using `llc`, the LLVM compiler
4. linking the object code for the task-graph and for the task variants
   into one executable using the `ld` linker.

The above-described flow has a notable implication on the implementation of the
CMake code. The rules that define Steps 3 and 4 cannot be formulated until the
rules of Step 2 run. This is because in order to link tasks together, it is
necessary to know what tasks there are, and that knowledge exists only
in the meta-program, at its runtime (the meta-program may generate tasks
when it runs, so it's not possible to avoid this dependency by somehow parsing
the meta-program code, even in principle). In order to account for this
dependency, rules for Step 3 and 4 are implemented in a separate nested
CMake project, that is generated by the parent CMake project at the end of Step
2 (after the meta-program has run and has output all the needed information
about tasks).

It would be feasible to build the functionality for constructing the
executable into the CASPER library linked with the meta-program (which
does all the other lowering steps). This might make for a simpler interface
for the user. However, the benefit of the present approach is that it
is more flexible: the user can change some aspects of this final linking
more easily by editting their application's CMake build code (that
calls CASPER-provided CMake code), as opposed to having to extend this
functionality in the CASPER Compiler, and rebuild the compiler.

The executable is ready to be run on the target platform (e.g. an HPC
cluster). For applications whose tasks support parallel execution using MPI,
the executable can be invoked using an MPI launcher (e.g. `mpirun` or `prun`).
Any task that is not MPI-capable will execute on each rank; each
tasks that is MPI-capable will get distributed among the reanks by
the logic private to that task's implementation (e.g. Distributed Halide,
Firedrake's MPI capability, etc).

Installation
------------

Exactly one system is supported: a specific snapshot of Gentoo Prefix (similar
to a container, but works without root privileges); but this one
Prefix system itself is portable to most Linux distributions, including on HPC
clusters, and though it, CASPER Compiler has been successfully tested on:
* OLCF Summit
* ANL Theta
* CentOS 7
* Ubuntu 18.04

The instructions for building Gentoo Prefix are in the
[casper-utils](https://github.com/ISI-apex/casper-utils.git) repository.

The package versions installed in the Gentoo Prefix snapshot are the
authoritative source for the dependency graph of CASPER Compiler. The
following information about dependencies serves a rough informational
purpose only:

* LLVM with MLIR components (tested with 11.0.1)
* Halide (tested with unreleased >11.0.1)
* Python 3 (tested with 3.8)
* Firedrake (tested with 20210226)
* PETSc (tested with unreleased >3.15)
* SLEPc (tested with unreleased >3.15)
* Solvers: HYPRE, MUMPS, SuperLU\_DIST, and others
* ...and, many many transitive dependencies of the above

### Build

The build instructions are also available in the `casper-utils.git`
repository linked above.

To build the compiler, enter the Gentoo Prefix and run:

    $ mkdir build && cd build
    $ CC=clang CXX=clang++ cmake -DCMAKE_BUILD_TYPE=Release ..
    $ make

The compiler can only be built with Clang, not with GCC.

The compiler may be used from its build directory or it may be installed into
the system with (untested):

    $ make install

#### Building applications

To build an application using the CASPER Compiler, the application
build must be defined using CMake. In the application's `CMakeLists.txt`
include the CASPER package:

    find_package(CASPER REQUIRED)

If the CASPER Compiler has not been installed into the system but only built,
then you may give a path to its source directory as a hint:

    find_package(CASPER REQUIRED PATHS "/path/to/casper-compiler/source)

Next, invoke the API function to create rules for a CASPER application
executable:

    casper_add_exec(sarbp sarbp.meta
      SOURCES sarbp.meta.cpp tasks.cpp
    )

The first argument is the name of the application executable, the second is the
name of the meta-program executable. The source files for the meta-program
and the source files that contain the generators for the are listed in
the `SOURCES` array.

There are several optional arguments to the `casper_add_exec` function:

* `C_KERNEL_SOURCES`: source files with non-DSL task functions written in C
* `EXTRA_INCLUDE_DIRS`: include paths to add when building the application
* `EXTRA_PACKAGES`: CMake packages to include when in the application project
   (it may be necessary to list `Halide` here, as a temporary workaround)
* `EXTRA_LIBRARIES`: libraries to add to the link line (may be CMake targets)
  (it may be necessary to list `Halide::Halide` here, as a temporary workaround)
* `PLATFORM`: platform definition file in `.ini` format, part of CASPER
  Autotuner integration (unused so far)
* `TUNED_PARAMS`: file in `.ini` format that defines the variants of
   task object code to generate for each task, by setting the values of
   generator parameters for each variant
* `RUNTIME_CFG`: runtime configuration in `.ini` format, that sets
   some parameters read by the CASPER Runtime when the application runs.

The CMake script for the application may add arbirary CMake directives, e.g.
add more libraries to link the meta-program with, define custom targets, etc.

Metaprogram API for Application
-------------------------------

Programs compiled by CASPER Compiler are graphs of tasks that are defined
by a meta-program. The meta-program is written in C++ and constructs the
task graph using CASPER Compiler API. This API is summarized here. This
is not an exhaustive reference, but serves as an "at-a-glance" description of
what CASPER Compiler is and does. Functional examples of applications
are given in the `casper-utils.git` repo in `exp/apps/casper/`.

A typical meta-program would look as follows:

    #include <casper.h>

    int main(int argc, char **argv) {
        Options opts;
        opts.parseOrExit(argc, argv);

        TaskGraph tg("blur");

        Dat *img = tg.createFloatDat(2, {IMG_HEIGHT, IMG_WIDTH});
        Dat *img_blurred = tg.createDat(IMG_HEIGHT, IMG_WIDTH);

        Task& task_load = tg.createTask(CKernel("hal_blur"), {img}, {});
        Task& task_blur = tg.createTask(HalideKernel("hal_blur"),
                            {img, img_blurred}, {&task_load});

        return tryCompile(tg, opts);
    }

The `TaskGraph` objects represents the application program. The nodes
in this graphs are tasks, and the edges are logical dependencies between
tasks. The task objects are instantiated using `createTask` method.
This `reateTask` method accepts a set of kernel objects:
* `HalideKernel`: a task written in the Halide DSL
* `FEMAKernel`: a task written with UFL expressions
* `CKernel`: a non-DSL task function written in C
* `PyKernel`: a non-DSL task function written in Python
The argument to the kernel constructors is the name of the entrypoint function
as a string.

The dependencies between tasks are created implicitly when a task
object is passed to a constructor of another task (in the list that is the
third argument to the task constructor).

The data buffers are declared using `createDat`. They are notionally
associated with the task graph, however they are neither nodes nor edges
in the graph. There is a family of `create*Dat` methods for data arrays
of different types, all of which take the number of dimensions as the
first argument:
* `createDoubleDat`: double-precision floating-point
* `createFloatDat`: single-precision floating-point
* `createIntDat`: integer, bitwidth is taken as an argument

It is also possible to create scalar values of different types to pass them
among tasks, with `create*Scalar` methods. The scalars are "boxed" and passed
by reference, so the task may modify a scalar value given to it.

For passing data and state between UFL and Python tasks, a generic
Python object is used as a container. Such an object can be created
with `createPyObj` and passed to tasks just like a data buffer.

Alongside the meta-program that defines the task graph, each task
is implemented (in separate source files) using a generator in
the respective DSL.

A Halide task implementation would look like the following:

    #include <Halide.h>

    using namespace Halide;
    using namespace Halide::Tools;

    class BlurGenerator : public Halide::Generator<BlurGenerator> {
    public:
        GeneratorParam<int> tile_x{"tile_x", 1};

        Input<Buffer<double>> input{"input", 2};
        Output<Buffer<double>> output{"blur_y", 2};
        
        void generate() {
            output(x) = 0.5 * intput(x) + 0.5 * input(x + 1);
        }

        void schedule() {
            output.compute_root();
        }
    }
    HALIDE_REGISTER_GENERATOR(HalideBlur, halide_blur)

The name of this task that is referenced in the metaprogram is the second
argument to `HALIDE_REGISTER_GENERATOR` macro: `halide_blur`.

A UFL task implementation would look lie the following:

    import casper
    from firedrake import *
    from firedrake.petsc import PETSc

    def generate():
        tasks = dict()
        mesh = UnitSquareMesh(mesh_size, mesh_size)

        V = FunctionSpace(mesh, "Lagrange", degree)
        ME = V*V
        du = TrialFunction(ME)
        q, v = TestFunctions(ME)
        u, u0 = Function(ME), Function(u0)
        ...

        F0 = c*q*dx - c0*q*dx + dt*dot(grad(mu_mid), grad(q))*dx
        F1 = mu*v*dx - dfdc*v*dx - lmbda*dot(grad(c), grad(v))*dx
        F = F0 + F1
        J = derivative(F, u, du)

        trial, test = TrialFunction(V), TestFunction(V)
        tasks["hats"] = casper.assemble(sqrt(a) * inner(trial, test)*dx + \
                sqrt(c)*inner(grad(trial), grad(test))*dx)
        tasks["assign"] = ...
        ...

        problem = NonlinearVariationalProblem(F, u, J=J)
        solver = NonlinearVariationalSolver(problem, solver_parameters=params)
        return tasks, solver, dict(u=u, u0=u0)

    def solve(ctx, state): # runs only at runtime
        hats = state["hats"] # reference to compiled kernel
        solver, u, u0 = ctx[1], ctx[2]["u"], ctx[2]["u0"] # runtime state

        ksp_hats = PETSc.KSP()
        ksp_hats.create()
        ksp_hats.setOperators(hats)
        ...

        for step in range(STEPS):
            casper.invoke_task(ctx, "assign", state)
            solver.solve()
        ...

The `generate` method defines the UFL expressions, each of which corresponds to
a kernel, which is going to be compiled when the meta-program runs. At
compile-time, the meta-program invokes this `generate` method, and then invokes
the UFL compiler on each returned UFL expression.  The `generate` method is
also invoked at runtime (see details in the Implementation section of this
document). The `solve` method is invoked only at runtime, it invokes compiled
kernels. This invocation takes one of two forms: (A) the kernel is passed to
`petsc4py`, or (B) the kernel is invoked explicitly with `casper.invoke_task`
API call.

### Configuration files

The CASPER Compiler consumes some configuration files in the `.ini` format:
* `tuned-params.ini`: defines the task variants; a variant is a version of
   object code compiled for the same task source code, and is defined by
   a set of values for generator parameters. The format of this file is as
   follows:

       [<taskname> <variant_id>]
       <param_name> = <param_value>

    For example, two variants of the blur task:

       [blur 0]
       target = x86-64-linux-no_runtime-sse41
       vectorsize = 16

       [blur 1]
       target = x86-64-linux-no_runtime-sse41
       vectorsize = 32

    The target parameter specifies the target architectrute for the Halide
    code-generator, and can be different for different variants (e.g. GPU
    variant and CPU variant).

* `crt.ini`: configuration parameters for CASPER Runtime. These are read
   by CASPER Runtime when the application executes. The format is a
   flat set of key-value pairs, for example, to tell the CASPER Runtime to
   always invoke the variant with id 1 (note: this should include a task
   name, but doesn't yet):

       variant_id = 1
