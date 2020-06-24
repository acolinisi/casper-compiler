#include "Build.h"
#include "TaskGraph.h"

#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace cac;

// Traverse the task graph in depth-first order
void invokeKernels(OpBuilder &builder, MLIRContext &context, Task& task)
{
  // Constant dat; TODO: support non-constant buffer

  for (Task *dep : task.deps) {
    if (!dep->visited)
      invokeKernels(builder, context, *dep);
  }

  // TODO: dats are re-used, so certainly wouldn't create for each task
  if (task.dat) { // TODO: invoke even if there is no dat attached
    Dat& dat = *task.dat;
    auto datTy = RankedTensorType::get({dat.rows, dat.cols},
	builder.getF64Type());
    auto datAttr = DenseElementsAttr::get(datTy, ArrayRef<double>(dat.vals));
    Value datIn = builder.create<toy::ConstantOp>(builder.getUnknownLoc(),
	datAttr);
    builder.create<toy::PrintOp>(builder.getUnknownLoc(), datIn);

    // Invoke the kernel
    auto funcAttr = StringAttr::get(StringRef(task.func), &context);
    auto kernOp = builder.create<toy::KernelOp>(builder.getUnknownLoc(),
		    datIn, funcAttr);

    builder.create<toy::PrintOp>(builder.getUnknownLoc(), datIn);
  } else {
  }
  task.visited = true;
}

int buildMLIRFromGraph(cac::TaskGraph &tg, MLIRContext &context,
    OwningModuleRef &module)
{
  module = OwningModuleRef(ModuleOp::create(
        UnknownLoc::get(&context)));

  OpBuilder builder(module->getBodyRegion());

  // create main() function
  auto mainTy = FunctionType::get({}, {builder.getI32Type()}, &context);
  FuncOp main = builder.create<FuncOp>(builder.getUnknownLoc(),
      StringRef("main", 4), mainTy,
      ArrayRef<NamedAttribute>{});
  auto &entryBlock = *main.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Invoke the kernel for each task.
  // We don't bother to keep track of actual roots, any (all) could be root.
  // Could track real roots; then have a root task, and no loop here.
  // For testing: process last-added task first.
  //for (auto rooti = tg.tasks.rbegin(); rooti != tg.tasks.rend(); ++rooti) {
  //  std::unique_ptr<Task>& root = *rooti;
  for (std::unique_ptr<Task>& root : tg.tasks) {
    if (!root->visited)
      invokeKernels(builder, context, *root);
  }

  // Return from main
  auto zeroVal = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(),
		  builder.getI32IntegerAttr(0));
  builder.create<ReturnOp>(builder.getUnknownLoc(),
		  ArrayRef<Value>{zeroVal});
  return 0;
}
