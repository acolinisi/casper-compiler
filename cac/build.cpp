#include "Build.h"
#include "Platform.h"
#include "KnowledgeBase.h"
#include "TaskGraph.h"
#include "TaskGraphImpl.h"

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

namespace {

// Traverse the task graph in depth-first order
void invokeKernels(OpBuilder &builder, MLIRContext &context, cac::Task& task,
    cac::Platform &plat, bool printDat)
{
  // Constant dat; TODO: support non-constant buffer

  for (cac::Task *dep : task.deps) {
    if (!dep->visited)
      invokeKernels(builder, context, *dep, plat, printDat);
  }

  // Invoke the kernel
  auto funcAttr = StringAttr::get(StringRef(task.func), &context);
  NamedAttribute funcNAttr(Identifier::get(StringRef("func"), &context),
      funcAttr);

  std::vector<Attribute> variants;
  for (auto &nodeDesc : plat.nodeTypes) {
    auto idAttr = IntegerAttr::get(builder.getI32Type(), nodeDesc.id);
    variants.push_back(idAttr);
  }
  auto variantsAttr = ArrayAttr::get(variants, &context);
  NamedAttribute variantsNAttr(
      Identifier::get(StringRef("variants"), &context), variantsAttr);

  std::vector<mlir::Value> args;
  for (cac::Value *val : task.args) {
    args.push_back(val->getImpl()->ref);
  }

  // We want TaskGraph to be decoupled from compilation implementation, so
  // we can't put these actions into a virtual method on the task objects.
  // TODO: switch to polymorphism, while still decoupling using impl ptr?
  switch (task.type) {
  case cac::Task::Halide:
      builder.create<toy::HalideKernelOp>(builder.getUnknownLoc(),
	ArrayRef<Type>{}, ValueRange(args),
	ArrayRef<NamedAttribute>{funcNAttr, variantsNAttr});
      break;
  case cac::Task::C:
      builder.create<toy::KernelOp>(builder.getUnknownLoc(),
	ArrayRef<Type>{}, ValueRange(args),
	ArrayRef<NamedAttribute>{funcNAttr, variantsNAttr});
      break;
  case cac::Task::Python: {
      cac::PyTask& pyTask = static_cast<cac::PyTask&>(task);
      auto pyModAttr = StringAttr::get(StringRef(pyTask.module), &context);
      NamedAttribute pyModNAttr(Identifier::get(StringRef("module"),
	    &context), pyModAttr);
      builder.create<toy::PyKernelOp>(builder.getUnknownLoc(),
	ArrayRef<Type>{}, ValueRange(args),
	ArrayRef<NamedAttribute>{pyModNAttr, funcNAttr, variantsNAttr});
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

mlir::LLVM::LLVMFuncOp declare_void_func(mlir::LLVM::LLVMDialect *llvmDialect,
    OpBuilder &builder, OwningModuleRef &module, StringRef func)
{
  auto llvmVoidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmVoidTy, {},
      /*isVarArg*/ false);
  return builder.create<LLVM::LLVMFuncOp>(module->getLoc(), func,
      llvmFnType);
}
mlir::LLVM::LLVMFuncOp declare_alloc_obj_func(mlir::LLVM::LLVMDialect *llvmDialect,
    OpBuilder &builder, OwningModuleRef &module, StringRef func)
{
  auto llvmVoidPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmVoidPtrTy, {},
      /*isVarArg*/ false);
  return builder.create<LLVM::LLVMFuncOp>(module->getLoc(), func,
      llvmFnType);
}
mlir::LLVM::LLVMFuncOp declare_free_obj_func(mlir::LLVM::LLVMDialect *llvmDialect,
    OpBuilder &builder, OwningModuleRef &module, StringRef func)
{
  auto llvmVoidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
  auto llvmVoidPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmVoidTy,
      {llvmVoidPtrTy}, /*isVarArg*/ false);
  return builder.create<LLVM::LLVMFuncOp>(module->getLoc(), func,
      llvmFnType);
}

std::string makeHalideArtifactName(const std::string &generator,
    const cac::NodeDesc &nodeDesc)
{
      return generator + "_v" + std::to_string(nodeDesc.id);
}

} // anon namespace

namespace cac {

// TODO: turn this into 'createHalideGenerators()', and store the
// generator object in the task object.
// Populates tunable parameter names list in Halide task objects
void introspectHalideParams(cac::TaskGraph &tg) {
  for (auto &task : tg.tasks) {
    if (task->type == cac::Task::Halide) {
      cac::HalideTask *halideTaskObj =
	static_cast<cac::HalideTask *>(task.get());
      const std::string &generator = task->func;
      halideTaskObj->params = cac::introspectHalideParams(generator);
    }
  }
}

int buildMLIRFromGraph(cac::TaskGraph &tg, cac::Platform &plat,
    cac::KnowledgeBase &kb,
    MLIRContext &context, OwningModuleRef &module)
{
  std::vector<std::string> generators;
  for (auto &task : tg.tasks) {
    if (task->type == cac::Task::Halide)
      generators.push_back(task->func);
  }

  for (auto& generator : generators) {
    // Compile as many variants as there are node types in the platform
    for (auto &nodeDesc : plat.nodeTypes) {
      const std::string &artifact =
	makeHalideArtifactName(generator, nodeDesc);
      auto& params = kb.getParams(generator, nodeDesc);

      std::cerr << "params for generator " << generator << ":" << std::endl;
      for (auto &kv : params) {
	std::cerr << kv.first << " = " << kv.second << std::endl;
      }

      cac::compileHalideKernel(generator, artifact, params);
    }
  }
  cac::compileHalideRuntime();

  module = OwningModuleRef(ModuleOp::create(
        UnknownLoc::get(&context)));

  OpBuilder builder(module->getBodyRegion());
  auto loc = builder.getUnknownLoc();
  auto llvmDialect = context.getRegisteredDialect<LLVM::LLVMDialect>();

  // Names are contract with Casper runtime
  const char *INIT_PY_FUNC = "init_python";
  const char *FIN_PY_FUNC = "finalize_python";
  const char *PY_ALLOC_OBJ_FUNC = "py_alloc_obj";
  const char *PY_FREE_OBJ_FUNC = "py_free_obj";

  auto initPyFunc = declare_void_func(llvmDialect, builder, module,
      INIT_PY_FUNC);
  auto finPyFunc = declare_void_func(llvmDialect, builder, module,
      FIN_PY_FUNC);
  auto pyAllocObjFunc = declare_alloc_obj_func(llvmDialect, builder, module,
      PY_ALLOC_OBJ_FUNC);
  auto pyFreeObjFunc = declare_free_obj_func(llvmDialect, builder, module,
      PY_FREE_OBJ_FUNC);

  // create main() function
  auto mainTy = FunctionType::get({}, {builder.getI32Type()}, &context);
  FuncOp main = builder.create<FuncOp>(loc,
      StringRef("main", 4), mainTy,
      ArrayRef<NamedAttribute>{});
  auto &entryBlock = *main.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  builder.create<LLVM::CallOp>(loc, initPyFunc, ValueRange{});

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
      auto elemTy = builder.getF64Type();
      auto memrefTy = MemRefType::get({datImpl->rows, datImpl->cols}, elemTy);
      datImpl->ref = builder.create<AllocOp>(loc, memrefTy);

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
      invokeKernels(builder, context, *root, plat, tg.datPrintEnabled);
  }

  builder.create<LLVM::CallOp>(loc, finPyFunc, ValueRange{});

  // Return from main
  auto zeroVal = builder.create<mlir::ConstantOp>(loc,
		  builder.getI32IntegerAttr(0));
  builder.create<ReturnOp>(loc, ArrayRef<mlir::Value>{zeroVal});
  return 0;
}

} // namespace cac
