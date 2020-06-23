#include "mlir/IR/Attributes.h"

#include "toy/Build.h"
#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "llvm/ADT/StringRef.h"

using namespace mlir;

int buildMLIR(MLIRContext &context,
              OwningModuleRef &module)
{
  module = OwningModuleRef(ModuleOp::create(
        UnknownLoc::get(&context)));

  OpBuilder builder(module->getBodyRegion());
  auto mainTy = FunctionType::get({}, {builder.getI32Type()}, &context);
  FuncOp main = builder.create<FuncOp>(builder.getUnknownLoc(),
      StringRef("main", 4), mainTy,
      ArrayRef<NamedAttribute>{});
  auto &entryBlock = *main.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  auto datTy = RankedTensorType::get({3, 2}, builder.getF64Type());
  auto datAttr = DenseElementsAttr::get(datTy, {
	1.000000e+00, -2.000000e+00, 3.000000e+00,
	4.000000e+00, 5.000000e+00, -6.000000e+00
  });
  Value datIn = builder.create<toy::ConstantOp>(builder.getUnknownLoc(),
		  datAttr);

  builder.create<toy::PrintOp>(builder.getUnknownLoc(), datIn);

  auto funcAttr = StringAttr::get(StringRef("mat_abs", 7), &context);
  auto kernOp = builder.create<toy::KernelOp>(builder.getUnknownLoc(),
		  datIn, funcAttr);

  builder.create<toy::PrintOp>(builder.getUnknownLoc(), datIn);

  auto zeroVal = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(),
		  builder.getI32IntegerAttr(0));
  builder.create<ReturnOp>(builder.getUnknownLoc(),
		  ArrayRef<Value>{zeroVal});
  return 0;
}
