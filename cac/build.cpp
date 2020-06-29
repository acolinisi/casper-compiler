#include "Build.h"
#include "TaskGraph.h"
#include "DatImpl.h"

#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

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
  if (task.dats.size() > 0) { // TODO: invoke even if there is no dat attached
    // Invoke the kernel
    auto funcAttr = StringAttr::get(StringRef(task.func), &context);
    NamedAttribute funcNAttr(Identifier::get(StringRef("func"), &context),
	funcAttr);

    std::vector<Value> args;
    for (Dat *dat : task.dats) {
      args.push_back(dat->impl->allocOp);
    }
    auto kernOp = builder.create<toy::KernelOp>(builder.getUnknownLoc(),
      ArrayRef<Type>{}, ValueRange(args), ArrayRef<NamedAttribute>{funcNAttr});

    for (Dat *dat : task.dats) {
      builder.create<toy::PrintOp>(builder.getUnknownLoc(), dat->impl->allocOp);
    }
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
  auto loc = builder.getUnknownLoc();

  // create main() function
  auto mainTy = FunctionType::get({}, {builder.getI32Type()}, &context);
  FuncOp main = builder.create<FuncOp>(loc,
      StringRef("main", 4), mainTy,
      ArrayRef<NamedAttribute>{});
  auto &entryBlock = *main.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // For now, we pre-allocate all data buffers, at the beginning of main()
  // and deallocate them at the end (we don't track lifetime of buffers).
  for (auto& dat : tg.dats) {
    auto elemTy = builder.getF64Type();
    auto memrefTy = MemRefType::get({dat->rows, dat->cols}, elemTy);
    dat->impl->allocOp = builder.create<AllocOp>(loc, memrefTy);

    // Load constant values if any were given
    if (dat->vals.size() > 0) {
      // Copied from lowering of ConstantOp

      // We will be generating constant indices up-to the largest dimension.
      // Create these constants up-front to avoid large amounts of redundant
      // operations.
      auto valueShape = memrefTy.getShape();
      SmallVector<Value, 8> constantIndices;

      if (!valueShape.empty()) {
	for (auto i : llvm::seq<int64_t>(
		0, *std::max_element(valueShape.begin(), valueShape.end())))
	 constantIndices.push_back(builder.create<ConstantIndexOp>(loc, i));
      } else {
	// This is the case of a tensor of rank 0.
	constantIndices.push_back(builder.create<ConstantIndexOp>(loc, 0));
      }
      // The constant operation represents a multi-dimensional constant, so we
      // will need to generate a store for each of the elements. The following
      // functor recursively walks the dimensions of the constant shape,
      // generating a store when the recursion hits the base case.
      SmallVector<Value, 2> indices;
      //auto valueIt = constantValue.getValues<FloatAttr>().begin();
      auto valueIt = dat->vals.begin();
      std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
	// The last dimension is the base case of the recursion, at this point
	// we store the element at the given index.
	if (dimension == valueShape.size()) {
	  auto floatAttr = FloatAttr::get(elemTy, *valueIt++);
	  builder.create<AffineStoreOp>(
	      loc, builder.create<ConstantOp>(loc, floatAttr),
	      dat->impl->allocOp,
	      llvm::makeArrayRef(indices));
	  return;
	}

	// Otherwise, iterate over the current dimension and add the indices to
	// the list.
	for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
	  indices.push_back(constantIndices[i]);
	  storeElements(dimension + 1);
	  indices.pop_back();
	}
      };

      // Start the element storing recursion from the first dimension.
      storeElements(/*dimension=*/0);
    }
  }

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
  auto zeroVal = builder.create<mlir::ConstantOp>(loc,
		  builder.getI32IntegerAttr(0));
  builder.create<ReturnOp>(loc, ArrayRef<Value>{zeroVal});
  return 0;
}
