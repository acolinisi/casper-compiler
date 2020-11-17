#include "Platform.h"
#include "KnowledgeBase.h"
#include "TaskGraph.h"
#include "TaskGraphImpl.h"

#include "toy/Dialect.h"
#include "toy/BuildHelpers.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

// Traverse the task graph in depth-first order
void invokeKernels(OpBuilder &builder, MLIRContext &context,
    mlir::LLVM::LLVMDialect *llvmDialect, cac::Task& task,
    std::vector<unsigned> variantIds, bool printDat, bool profilingHarness,
    mlir::LLVM::LLVMFuncOp pyAllocStrFunc)
{
  // Constant dat; TODO: support non-constant buffer

  for (cac::Task *dep : task.deps) {
    if (!dep->visited)
      invokeKernels(builder, context, llvmDialect, *dep, variantIds, printDat,
	  profilingHarness, pyAllocStrFunc);
  }

  // Invoke the kernel
  auto funcId = Identifier::get(StringRef("func"), &context);
  auto funcAttr = StringAttr::get(StringRef(task.func), &context);
  NamedAttribute funcNAttr(funcId, funcAttr);

  std::vector<Attribute> variants;
  for (auto &variantId : variantIds) {
    auto idAttr = IntegerAttr::get(builder.getI32Type(), variantId);
    variants.push_back(idAttr);
  }
  auto variantsAttr = ArrayAttr::get(variants, &context);
  NamedAttribute variantsNAttr(
      Identifier::get(StringRef("variants"), &context), variantsAttr);

  std::vector<mlir::Value> args;
  for (cac::Value *val : task.args) {
    args.push_back(val->getImpl()->ref);
  }

  auto loc = builder.getUnknownLoc();

  // We want TaskGraph to be decoupled from compilation implementation, so
  // we can't put these actions into a virtual method on the task objects.
  // TODO: switch to polymorphism, while still decoupling using impl ptr?
  switch (task.type) {
  case cac::Task::Halide: {
      auto profileAttr = IntegerAttr::get(builder.getIntegerType(32),
	  profilingHarness);
      NamedAttribute profileNAttr(Identifier::get(StringRef("profile"),
	    &context), profileAttr);
      builder.create<toy::HalideKernelOp>(loc,
	ArrayRef<Type>{}, ValueRange(args),
	ArrayRef<NamedAttribute>{funcNAttr, variantsNAttr, profileNAttr});
      break;
  }
  case cac::Task::C:
      builder.create<toy::KernelOp>(builder.getUnknownLoc(),
	ArrayRef<Type>{}, ValueRange(args),
	ArrayRef<NamedAttribute>{funcNAttr, variantsNAttr});
      break;
  case cac::Task::Python: {
      cac::PyTask& pyTask = static_cast<cac::PyTask&>(task);
      std::vector<NamedAttribute> attrs;

      attrs.push_back(variantsNAttr);

      std::vector<mlir::Value> pyArgs{pyTask.impl->generatorContext};

      if (pyTask.type == cac::PyTask::Generated) {
	cac::PyGenedTask& pyGenedTask =
	  static_cast<cac::PyGenedTask&>(pyTask);

	auto pyModAttr = StringAttr::get(StringRef("casper"), &context);
	NamedAttribute pyModNAttr(Identifier::get(StringRef("module"),
	      &context), pyModAttr);
	attrs.push_back(pyModNAttr);

	auto kernelAttr = StringAttr::get(StringRef(pyGenedTask.kernel),
	    &context);
	NamedAttribute kernelNAttr(Identifier::get(StringRef("kernel"),
	      &context), kernelAttr);
	attrs.push_back(kernelNAttr);

	// helper function name is in casper.py
	auto pyFuncAttr = StringAttr::get(StringRef("invoke_task"),
	    &context);
	NamedAttribute pyFuncNAttr(funcId, pyFuncAttr);
	attrs.push_back(pyFuncNAttr);

	mlir::Value kernelNameStr = toy::allocString(builder, llvmDialect,
	  loc, pyGenedTask.kernel);
	auto callOp = builder.create<LLVM::CallOp>(loc, pyAllocStrFunc,
	    ValueRange{kernelNameStr});
	assert(callOp.getNumResults() == 1);
	mlir::Value kernelNamePyStr = callOp.getResult(0);

	pyArgs.push_back(kernelNamePyStr);

      } else {
	auto pyModAttr = StringAttr::get(StringRef(pyTask.module), &context);
	NamedAttribute pyModNAttr(Identifier::get(StringRef("module"),
	      &context), pyModAttr);
	attrs.push_back(pyModNAttr);
	attrs.push_back(funcNAttr);
      }

      for (auto &arg : args) {
	pyArgs.push_back(arg);
      }

      builder.create<toy::PyKernelOp>(builder.getUnknownLoc(),
	ArrayRef<Type>{}, ValueRange(pyArgs), attrs);
      break;
  }
  default:
    // TODO: figure out failure propagation
    assert("unsupported task type");
  }

  if (printDat) {
    for (cac::Value *val : task.args) {
      auto valImpl = val->getImpl();
      if (valImpl->type == cac::ValueImpl::Dat)
	builder.create<toy::PrintOp>(builder.getUnknownLoc(), valImpl->ref);
      // TODO: print for scalars
    }
  }
  task.visited = true;
}

mlir::LLVM::LLVMFuncOp declareVoidFunc(OpBuilder &builder,
    OwningModuleRef &module, mlir::LLVM::LLVMDialect *llvmDialect,
    StringRef func)
{
  auto llvmVoidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmVoidTy, {},
      /*isVarArg*/ false);
  return builder.create<LLVM::LLVMFuncOp>(module->getLoc(), func,
      llvmFnType);
}
mlir::LLVM::LLVMFuncOp declareRetPyObjFunc(OpBuilder &builder,
    OwningModuleRef &module, mlir::LLVM::LLVMDialect *llvmDialect,
    StringRef func)
{
  auto llvmVoidPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmVoidPtrTy, {},
      /*isVarArg*/ false);
  return builder.create<LLVM::LLVMFuncOp>(module->getLoc(), func,
      llvmFnType);
}
mlir::LLVM::LLVMFuncOp declareTakePyObjFunc(OpBuilder &builder,
    OwningModuleRef &module, mlir::LLVM::LLVMDialect *llvmDialect,
    StringRef func)
{
  auto llvmVoidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
  auto llvmVoidPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmVoidTy,
      {llvmVoidPtrTy}, /*isVarArg*/ false);
  return builder.create<LLVM::LLVMFuncOp>(module->getLoc(), func,
      llvmFnType);
}

mlir::LLVM::LLVMFuncOp declareTakeStrRetPyObjFunc(OpBuilder &builder,
    OwningModuleRef &module, mlir::LLVM::LLVMDialect *llvmDialect,
    StringRef func)
{
  auto llvmVoidPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
  auto llvmCharPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmCharPtrTy,
      {llvmVoidPtrTy}, /*isVarArg*/ false);
  return builder.create<LLVM::LLVMFuncOp>(module->getLoc(), func,
      llvmFnType);
}

mlir::LLVM::LLVMFuncOp declareInitProfilingFunc(OpBuilder &builder,
    OwningModuleRef &module, mlir::LLVM::LLVMDialect *llvmDialect)
{
  auto llvmVoidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
  auto llvmCharPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmVoidTy,
      {llvmCharPtrTy}, /*isVarArg*/ false);
  return builder.create<LLVM::LLVMFuncOp>(module->getLoc(), "_crt_prof_init",
      llvmFnType);
}

} // anon namespace

namespace cac {

int buildMLIRFromGraph(OwningModuleRef &module, cac::TaskGraph &tg,
    std::vector<unsigned> variantIds, MLIRContext &context,
    bool profilingHarness, const std::string &profilingMeasurementsFile)
{
  module = OwningModuleRef(ModuleOp::create(
        UnknownLoc::get(&context)));

  OpBuilder builder(module->getBodyRegion());
  auto loc = builder.getUnknownLoc();
  auto llvmDialect = context.getRegisteredDialect<LLVM::LLVMDialect>();

  // Names are contract with Casper runtime
  auto initPyFunc = declareVoidFunc(builder, module, llvmDialect,
      "_crt_py_init");
  auto finPyFunc = declareVoidFunc(builder, module, llvmDialect,
      "_crt_py_finalize");
  auto pyAllocObjFunc = declareRetPyObjFunc(builder, module, llvmDialect,
      "_crt_py_alloc_obj");
  auto pyFreeObjFunc = declareTakePyObjFunc(builder, module, llvmDialect,
      "_crt_py_free_obj");
  auto pyAllocStrFunc = declareTakeStrRetPyObjFunc(builder, module,
      llvmDialect, "_crt_py_alloc_str");
  auto pyConstructKernelsFunc = declareRetPyObjFunc(builder, module,
      llvmDialect, "_crt_py_construct_kernels");

  LLVM::LLVMFuncOp initProfFunc, finProfFunc;
  if (profilingHarness) {
    initProfFunc = declareInitProfilingFunc(builder, module, llvmDialect);
    finProfFunc = declareVoidFunc(builder, module, llvmDialect,
	"_crt_prof_finalize");
  }

  // create main() function
  auto mainTy = FunctionType::get({}, {builder.getI32Type()}, &context);
  FuncOp main = builder.create<FuncOp>(loc,
      StringRef("main", 4), mainTy,
      ArrayRef<NamedAttribute>{});
  auto &entryBlock = *main.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  builder.create<LLVM::CallOp>(loc, initPyFunc, ValueRange{});

  // TODO: should be conditional, also each task should be associated with
  // its generator, and there should be a context per generator, passed
  // to each task generated by that generator.
  auto pyConstructKernCallOp = builder.create<LLVM::CallOp>(loc,
      pyConstructKernelsFunc, ValueRange{});
  assert(pyConstructKernCallOp.getNumResults() == 1);
  auto pyGenContext = pyConstructKernCallOp.getResult(0);
  for (auto& task : tg.tasks) {
    if (task->type == Task::Python) {
      cac::PyTask& pyTask = static_cast<cac::PyTask&>(*task);
      pyTask.impl->generatorContext = pyGenContext;
    }
  }

  if (profilingHarness) {
    assert(profilingMeasurementsFile.size());
    mlir::Value profMeasFile = toy::allocString(builder, llvmDialect, loc,
	profilingMeasurementsFile);
    builder.create<LLVM::CallOp>(loc, initProfFunc, ValueRange{profMeasFile});
  }

  // For now, we pre-allocate all data buffers, at the beginning of main()
  // and deallocate them at the end (we don't track lifetime of buffers).
  for (auto& val : tg.values) {
    // TODO: switch to polymorphism in Impl types? each can allocate itself
    switch (val->getImpl()->type) {
    case cac::ValueImpl::Scalar: {
      cac::Scalar *scalar = static_cast<cac::Scalar*>(val.get());
      auto scalarImpl = scalar->getImpl();
      // Allocate on the stack, pass to kernel by value.
      // TODO: stay in std dialect! Have to go down to LLVM types because the
      // alloca in MLIR std dialect is for memrefs only (?).
      auto sTy = scalarImpl->getLLVMType(llvmDialect);
      auto sStdTy = scalarImpl->getType(builder);
      // TODO: assumes IndexType is 64-bit
      auto idxTy = LLVM::LLVMType::getInt64Ty(llvmDialect);
      mlir::Value one = builder.create<LLVM::ConstantOp>(loc, idxTy,
	  builder.getIntegerAttr(builder.getIndexType(), 1));
      auto sPtr = builder.create<LLVM::AllocaOp>(loc, sTy.getPointerTo(),
	  one, /*align=*/ 0);

      if (scalarImpl->initialized) {
	auto initVal = builder.create<LLVM::ConstantOp>(loc, sTy,
	    scalarImpl->getInitValue(builder));
	builder.create<LLVM::StoreOp>(loc, initVal, sPtr);
      }
      auto loaded = builder.create<LLVM::LoadOp>(loc, sPtr);
      // TODO: It's ridiculous to go LLVM->std here, we should stay in std, but
      // how to allocate a scalar in std? (alloca is only for memref? wtf)
      scalarImpl->ref = builder.create<LLVM::DialectCastOp>(loc,
	  sStdTy, loaded);
      break;
    }
    case cac::ValueImpl::Dat: {
      cac::Dat *dat = static_cast<cac::Dat*>(val.get());
      auto datImpl = dat->getImpl();
      auto elemTy = datImpl->getElementType(builder);
      auto memrefTy = MemRefType::get(
	  {datImpl->size[0], datImpl->size[1]}, elemTy);
      datImpl->ref = builder.create<AllocOp>(loc, memrefTy);

#if 0 // TODO: temporarily unsupporting const values
      // Load constant values if any were given
      if (datImpl->vals.size() > 0) {
	// Copied from lowering of ConstantOp

	// We will be generating constant indices up-to the largest dimension.
	// Create these constants up-front to avoid large amounts of redundant
	// operations.
	auto valueShape = memrefTy.getShape();
	SmallVector<mlir::Value, 8> constantIndices;

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
	SmallVector<mlir::Value, 2> indices;
	//auto valueIt = constantValue.getValues<FloatAttr>().begin();
	auto valueIt = datImpl->vals.begin();
	std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
	  // The last dimension is the base case of the recursion, at this point
	  // we store the element at the given index.
	  if (dimension == valueShape.size()) {
	    auto floatAttr = FloatAttr::get(elemTy, *valueIt++);
	    builder.create<AffineStoreOp>(
		loc, builder.create<ConstantOp>(loc, floatAttr),
		datImpl->ref, llvm::makeArrayRef(indices));
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
#endif
      break;
    }
    case cac::ValueImpl::PyObj: {
      cac::PyObj *pyObj = static_cast<cac::PyObj*>(val.get());
      auto pyObjImpl = pyObj->getImpl();
      auto callOp = builder.create<LLVM::CallOp>(loc, pyAllocObjFunc,
	  ValueRange{});
      assert(callOp.getNumResults() == 1);
      pyObjImpl->ref = callOp.getResult(0);
      // TODO: cleanup at end of main
      break;
    }
    default:
      // TODO: figure out error reporting
      assert("unsupported value type");
      return 1;
    }
  }

  // Invoke the kernel for each task.
  // We don't bother to keep track of actual roots, any (all) could be root.
  // Could track real roots; then have a root task, and no loop here.
  // For testing: process last-added task first.
  //for (auto rooti = tg.tasks.rbegin(); rooti != tg.tasks.rend(); ++rooti) {
  //  std::unique_ptr<Task>& root = *rooti;
  for (std::unique_ptr<cac::Task>& root : tg.tasks) {
    if (!root->visited)
      invokeKernels(builder, context, llvmDialect, *root, variantIds,
	  tg.datPrintEnabled, profilingHarness, pyAllocStrFunc);
  }

  if (profilingHarness) {
    builder.create<LLVM::CallOp>(loc, finProfFunc, ValueRange{});
  }

  builder.create<LLVM::CallOp>(loc, finPyFunc, ValueRange{});

  // Return from main
  auto zeroVal = builder.create<mlir::ConstantOp>(loc,
		  builder.getI32IntegerAttr(0));
  builder.create<ReturnOp>(loc, ArrayRef<mlir::Value>{zeroVal});
  return 0;
}

} // namespace cac
