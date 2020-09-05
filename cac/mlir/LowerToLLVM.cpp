//====- LowerToLLVM.cpp - Lowering from Toy+Affine+Std to LLVM ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements full lowering of Toy operations to LLVM MLIR dialect.
// 'toy.print' is lowered to a loop nest that calls `printf` on each element of
// the input array. The file also sets up the ToyToLLVMLoweringPass. This pass
// lowers the combination of Affine + SCF + Standard dialects to the LLVM one:
//
//                         Affine --
//                                  |
//                                  v
//                                  Standard --> LLVM (Dialect)
//                                  ^
//                                  |
//     'toy.print' --> Loop (SCF) --
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "toy/BuildHelpers.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include <string>
#include <iostream> // for debugging only

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {

Value createConstInt32(ConversionPatternRewriter &rewriter,
		TypeConverter *typeConverter, Location loc, int32_t val) {
	auto ty = rewriter.getI32Type();
	auto loweredTy = typeConverter->convertType(ty);
	auto attr = rewriter.getIntegerAttr(ty, val);
	return rewriter.create<LLVM::ConstantOp>(loc, loweredTy, attr);
}
Value createConstIndex(ConversionPatternRewriter &rewriter,
		TypeConverter *typeConverter, Location loc, unsigned val) {
	auto ty = rewriter.getIndexType();
	auto loweredTy = typeConverter->convertType(ty);
	auto attr = rewriter.getIntegerAttr(ty, val);
	return rewriter.create<LLVM::ConstantOp>(loc, loweredTy, attr);
}

void declareFunc(ConversionPatternRewriter &rewriter, ModuleOp &mod,
		StringRef funcName, LLVM::LLVMType &funcType) {
	// Insert the function declaration into the body of the parent module.
	PatternRewriter::InsertionGuard insertGuard(rewriter);
	rewriter.setInsertionPointToStart(mod.getBody());
	rewriter.create<LLVM::LLVMFuncOp>(mod.getLoc(), funcName, funcType);
}

LLVM::CallOp callCRT(ConversionPatternRewriter &rewriter, ModuleOp &mod,
		Location loc, StringRef funcName,
		std::function<LLVM::LLVMType()> typeCtor,
		ValueRange args = ValueRange{}) {
	if (!mod.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
		auto funcType = typeCtor();
		declareFunc(rewriter, mod, funcName, funcType);
	}
	auto funcOp = mod.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
	return rewriter.create<LLVM::CallOp>(loc, funcOp, args);
}

LLVM::CallOp callCRTStopWatchStart(ConversionPatternRewriter &rewriter,
		ModuleOp &mod, LLVM::LLVMDialect *llvmDialect, Location loc) {
	auto typeCtor = [llvmDialect]() {
		auto llvmVoidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
		return LLVM::LLVMType::getFunctionTy(llvmVoidTy, {}, /*VarArg*/ false);
	};
	return callCRT(rewriter, mod, loc, "_crt_prof_stopwatch_start", typeCtor);
}
LLVM::CallOp callCRTStopWatchStop(ConversionPatternRewriter &rewriter,
		ModuleOp &mod, LLVM::LLVMDialect *llvmDialect, Location loc) {
	auto typeCtor = [llvmDialect]() {
		auto llvmFloatTy = LLVM::LLVMType::getFloatTy(llvmDialect);
		return LLVM::LLVMType::getFunctionTy(llvmFloatTy, {}, /*VarArg*/ false);
	};
	return callCRT(rewriter, mod, loc, "_crt_prof_stopwatch_stop", typeCtor);
}
LLVM::CallOp callCRTLogMeasurement(ConversionPatternRewriter &rewriter,
		ModuleOp &mod, LLVM::LLVMDialect *llvmDialect, Location loc,
		ValueRange args) {
	auto typeCtor = [llvmDialect]() {
		auto llvmFloatTy = LLVM::LLVMType::getFloatTy(llvmDialect);
		auto llvmCharTy = LLVM::LLVMType::getInt8Ty(llvmDialect);
		auto llvmVoidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
		auto llvmI32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
        return LLVM::LLVMType::getFunctionTy(llvmVoidTy,
            {llvmCharTy.getPointerTo(), llvmI32Ty, llvmFloatTy},
            /*isVarArg*/ false);
	};
	return callCRT(rewriter, mod, loc, "_crt_prof_log_measurement", typeCtor,
			args);
}
LLVM::CallOp callCRTGetNodeTypeId(ConversionPatternRewriter &rewriter,
		ModuleOp &mod, LLVM::LLVMDialect *llvmDialect, Location loc) {
	auto typeCtor = [llvmDialect]() {
		auto llvmIntTy = LLVM::LLVMType::getInt32Ty(llvmDialect);
		return LLVM::LLVMType::getFunctionTy(llvmIntTy, {}, /*isVarArg*/ false);
	};
	return callCRT(rewriter, mod, loc, "_crt_plat_get_node_type_id", typeCtor);
}


/// Return a symbol reference to the kernel function, inserting it into the
/// module if necessary.
FlatSymbolRefAttr getOrInsertKernFunc(PatternRewriter &rewriter,
		ModuleOp module, StringRef name, LLVM::LLVMType llvmFnType) {
	auto *context = module.getContext();
	if (!module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
		// Insert the printf function into the body of the parent module.
		PatternRewriter::InsertionGuard insertGuard(rewriter);
		rewriter.setInsertionPointToStart(module.getBody());
		rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, llvmFnType);
	}
	return SymbolRefAttr::get(name, context);
}

// Allocate an indirection jump table: node type id -> kernel function pointer
class JumpTable {
public:
	JumpTable(ConversionPatternRewriter &rewriter, ModuleOp &mod,
		TypeConverter *typeConverter, Location loc,
		LLVM::LLVMType funcType, unsigned numEntries,
		std::function<std::pair<unsigned, FlatSymbolRefAttr>(unsigned)> map)
	: rewriter(rewriter), funcType(funcType) {
		Value numEntriesVal = createConstIndex(rewriter, typeConverter, loc,
				numEntries);
		tableAddr = rewriter.create<LLVM::AllocaOp>(loc,
				funcType.getPointerTo().getPointerTo(), numEntriesVal,
				/*alignment=*/0);

		for (int i = 0; i < numEntries; ++i) {
			std::pair<unsigned, FlatSymbolRefAttr> mapEntry = map(i);
			unsigned slotIdx = mapEntry.first;
			FlatSymbolRefAttr kernRef = mapEntry.second;

			// Get the pointer to the function variant
			auto funcAddr = rewriter.create<LLVM::AddressOfOp>(loc,
					funcType.getPointerTo(), kernRef);

			// Add the function pointer to a slot in the indirection jump table
			Value slotIdxVal = createConstIndex(rewriter, typeConverter, loc,
					slotIdx);
			auto slotAddr = rewriter.create<LLVM::GEPOp>(loc,
					funcType.getPointerTo(), tableAddr, ValueRange{slotIdxVal});
			rewriter.create<LLVM::StoreOp>(loc, funcAddr, slotAddr);
		}
	}
	Value lookup(Value index, Location loc) {
		auto slotAddr = rewriter.create<LLVM::GEPOp>(loc,
				funcType.getPointerTo(), tableAddr, ValueRange{index});
		return rewriter.create<LLVM::LoadOp>(loc, slotAddr);
	}
private:
	ConversionPatternRewriter &rewriter;
	LLVM::LLVMType funcType;
	Value tableAddr;
};

/// Lowers `toy.print` to a loop nest calling `printf` on each of the individual
/// elements of the array.
class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(toy::PrintOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto memRefType = (*op->operand_type_begin()).cast<MemRefType>();
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();
    auto *llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule, llvmDialect);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule,
        llvmDialect);
    Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule, llvmDialect);

    // Create a loop for each of the dimensions within the shape.
    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
      auto upperBound = rewriter.create<ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : *loop.getBody())
        rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());

      // Insert a newline after each of the inner dimensions of the shape.
      if (i != e - 1)
        rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                                newLineCst);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Generate a call to printf for the current element of the loop.
    auto printOp = cast<toy::PrintOp>(op);
    auto elementLoad = rewriter.create<LoadOp>(loc, printOp.input(), loopIvs);
    rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                            ArrayRef<Value>({formatSpecifierCst, elementLoad}));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module,
                                             LLVM::LLVMDialect *llvmDialect) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get("printf", context);

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
    auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmI32Ty, llvmI8PtrTy,
                                                    /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get("printf", context);
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module,
                                       LLVM::LLVMDialect *llvmDialect) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMType::getArrayTy(
          LLVM::LLVMType::getInt8Ty(llvmDialect), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(
        loc, LLVM::LLVMType::getInt64Ty(llvmDialect),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMType::getInt8PtrTy(llvmDialect), globalPtr,
        ArrayRef<Value>({cst0, cst0}));
  }
};

// Lowers a kernel invocation into a call to the kernel function
class KernelOpLowering : public ConversionPattern {
public:
  explicit KernelOpLowering(MLIRContext *context)
      : ConversionPattern(toy::KernelOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto memRefType = (*op->operand_type_begin()).cast<MemRefType>();
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();
    auto *llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get a symbol reference to the kernel function, inserting it if necessary.
    StringAttr funcAttr = op->getAttrOfType<StringAttr>("func");
    assert(funcAttr); // verified
    StringRef func = funcAttr.getValue();

    auto kernRef = getOrInsertKernFunc(rewriter, parentModule,
        func, op->getNumOperands(), llvmDialect);
    auto kernOp = cast<toy::KernelOp>(op);

    rewriter.create<CallOp>(loc, kernRef, ArrayRef<Type>(), kernOp.input());

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Return a symbol reference to the kernel function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertKernFunc(PatternRewriter &rewriter,
                                             ModuleOp module,
                                             StringRef name, int numOpers,
                                             LLVM::LLVMDialect *llvmDialect) {
    auto *context = module.getContext();
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {

      // Create a function declaration for the kernel, the signature is:
      //    `void (<unpacked MemRef arg>, ...)`
      auto llvmVoidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
      auto llvmITy = LLVM::LLVMType::getInt64Ty(llvmDialect);
      auto llvmDTy = LLVM::LLVMType::getDoubleTy(llvmDialect);

      std::vector<LLVM::LLVMType> opers;
      for (int i = 0; i < numOpers; ++i) {
        opers.push_back(llvmDTy.getPointerTo()); /* buffer */
        opers.push_back(llvmDTy.getPointerTo()); /* start of aligned data */
        opers.push_back(llvmITy); /* offset into buffer */
        opers.push_back(llvmITy); /* size per dim */
        opers.push_back(llvmITy);
        opers.push_back(llvmITy); /* stride per dim */
        opers.push_back(llvmITy);
      }

      // TODO: unranked memref?
      auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmVoidTy,
        opers, /*isVarArg*/ false);

      // Insert the function declaration into the body of the parent module.
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, llvmFnType);
    }
    return SymbolRefAttr::get(name, context);
  }
};

// Lowers a kernel invocation into a call to the kernel function
class PyKernelOpLowering : public ConversionPattern {
private:
  // Launcher function name is a contract with Casper runtime
  static constexpr char* LAUNCH_PY_FUNC = "launch_python";

public:
  explicit PyKernelOpLowering(TypeConverter &typeConverter,
      MLIRContext *context)
      : ConversionPattern(toy::PyKernelOp::getOperationName(), 1,
          typeConverter, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
#if 0
    auto memRefType = (*op->operand_type_begin()).cast<MemRefType>();
    auto memRefShape = memRefType.getShape();
#endif
    auto loc = op->getLoc();
    auto *llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Get a symbol reference to the kernel function, inserting it if necessary.
    // Declare launcher function: func name is a contract with Casper runtime

    auto llvmVoidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
    // TODO: implicit assumption that rewriter.getIndexType() is 64-bit
    auto llvmSizeTy = LLVM::LLVMType::getInt64Ty(llvmDialect);
    auto llvmPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
    auto llvmPtrArrTy = LLVM::LLVMType::getArrayTy(llvmPtrTy,
        operands.size());

    std::string launchPyFunc{LAUNCH_PY_FUNC};
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(launchPyFunc)) {
      auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmVoidTy,
          {/* module */ llvmPtrTy, /* func */ llvmPtrTy,
          /* num_args */ llvmSizeTy, /* args */ llvmPtrArrTy.getPointerTo()},
          /*isVarArg*/ false);

      // Insert the function declaration into the body of the parent module.
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), launchPyFunc, llvmFnType);
    }
    auto launchFuncRef = SymbolRefAttr::get(launchPyFunc, module.getContext());

    // Allocate char[] array and fill with value of attribute
    Value pyModNameArr = toy::allocString(rewriter, typeConverter, llvmDialect,
        loc, op, "module");
    Value pyFuncNameArr = toy::allocString(rewriter, typeConverter, llvmDialect,
        loc, op, "func");

    auto kernOp = cast<toy::PyKernelOp>(op);

    // Allocate array of args to the kernel launcher function

    Value zero = rewriter.create<LLVM::ConstantOp>(loc,
        typeConverter->convertType(rewriter.getIndexType()),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    Value one = rewriter.create<LLVM::ConstantOp>(loc,
        typeConverter->convertType(rewriter.getIndexType()),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    Value argsArrPtr = rewriter.create<LLVM::AllocaOp>(loc,
        llvmPtrArrTy.getPointerTo(), one, /*alignment=*/0);

    for (int index = 0; index < kernOp.input().size(); ++index) {
      auto indexVal = rewriter.create<LLVM::ConstantOp>(loc,
          typeConverter->convertType(rewriter.getIndexType()),
          rewriter.getIntegerAttr(rewriter.getIndexType(), index));

      auto elemAddr = rewriter.create<LLVM::GEPOp>(loc,
          llvmPtrTy.getPointerTo(),
          argsArrPtr, ValueRange{zero, indexVal});

      rewriter.create<LLVM::StoreOp>(loc, kernOp.input()[index], elemAddr);
    }

    Value argsLen = rewriter.create<LLVM::ConstantOp>(loc,
        typeConverter->convertType(rewriter.getIndexType()),
        rewriter.getIntegerAttr(rewriter.getIndexType(),
          kernOp.input().size()));

    SmallVector<Value, 8> args{pyModNameArr, pyFuncNameArr,
                               argsLen, argsArrPtr};

    rewriter.create<CallOp>(loc, launchFuncRef, ArrayRef<Type>(), args);

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
  }

private:
};

class ToyLLVMTypeConverter : public LLVMTypeConverter {
public:
  ToyLLVMTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx) {
      addConversion([&](MemRefType type) {
          return convertMemRefTypeToStructHalideBuf(type);
      });
    }
  Type convertMemRefTypeToStructHalideBuf(MemRefType type);
};

// Extract an LLVM IR type from the LLVM IR dialect type.
static LLVM::LLVMType unwrap(Type type) {
  if (!type)
    return nullptr;
  auto *mlirContext = type.getContext();
  auto wrappedLLVMType = type.dyn_cast<LLVM::LLVMType>();
  if (!wrappedLLVMType)
    emitError(UnknownLoc::get(mlirContext),
              "conversion resulted in a non-LLVM type");
  return wrappedLLVMType;
}

Type ToyLLVMTypeConverter::convertMemRefTypeToStructHalideBuf(MemRefType type) {
  LLVM::LLVMType elementType = unwrap(convertType(type.getElementType()));
  if (!elementType)
    return {};

  LLVM::LLVMDialect *llvmDialect = &elementType.getDialect();

  auto memSpace = type.getMemorySpace();
  auto int64Ty = LLVM::LLVMType::getInt64Ty(llvmDialect);
  auto int32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto int16Ty = LLVM::LLVMType::getInt16Ty(llvmDialect);
  auto int8Ty = LLVM::LLVMType::getInt8Ty(llvmDialect);
  auto voidPtrTy = int8Ty.getPointerTo(memSpace);

  auto typeTy = LLVM::LLVMType::getStructTy(
      int8Ty, // uint8_t/halide_type_code_t code
      int8Ty, // uint8_t bits
      int16Ty // int16_t lanes
  );

  auto dimTy = LLVM::LLVMType::getStructTy(
      int32Ty, // int32_t min
      int32Ty, // int32_t extent
      int32Ty, // int32_t stride
      int32Ty  // uint32_t flags
  );

  return LLVM::LLVMType::getStructTy(
      // See struct halide_buffer_t in Halide/src/rutnime/HalideRuntime.h
      int64Ty,   // uint64_t device
      voidPtrTy, // struct halide_device_interface_t *device_interface
      int8Ty.getPointerTo(memSpace), // uint8_t *host
      int64Ty,   // uint64_t flags
      typeTy,   // struct halide_type_t type
      int32Ty,   // int32_t dimensions
      dimTy.getPointerTo(memSpace), // struct halide_dimension_t *dim
      voidPtrTy  // void *padding
    );
}

// Lowers a Halide kernel invocation into a call to the kernel function
class HalideKernelOpLowering : public ConversionPattern {
private:
  // Field offsets from struct halide_buffer_t in HalideRuntime.h
  static const int kHalideBuffDevice = 0;
  static const int kHalideBuffDeviceInterface = 1;
  static const int kHalideBuffHost = 2;
  static const int kHalideBuffFlags = 3;
  static const int kHalideBuffType = 4;
  static const int kHalideBuffDimensions = 5;
  static const int kHalideBuffDim = 6;
  static const int kHalideBuffPadding = 7;

  static const int kHalideBuffTypeCode = 0;
  static const int kHalideBuffTypeBits = 1;
  static const int kHalideBuffTypeLanes = 2;

  static const int kHalideBuffDimMin = 0;
  static const int kHalideBuffDimExtent = 1;
  static const int kHalideBuffDimStride = 2;
  static const int kHalideBuffDimFlags = 3;

  // enum halide_type_code_t in HalideRuntime.h
  static const int kHalideTypeInt = 0;     // signed integers
  static const int kHalideTypeUInt = 1;    // unsigned integers
  static const int kHalideTypeFloat = 2;   // IEEE floating point numbers
  static const int kHalideTypeHandle = 3;  // opaque pointer type (void *)
  static const int kHalideTypeBFloat = 4;  // floating point numbers in the bfloat format

public:
  explicit HalideKernelOpLowering(TypeConverter &typeConverter,
      MLIRContext *context)
      : ConversionPattern(toy::HalideKernelOp::getOperationName(), 1,
          typeConverter, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    toy::HalideKernelOp opx = llvm::dyn_cast<toy::HalideKernelOp>(op);

    auto loc = op->getLoc();
    auto *llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // MemRefType -> struct halide_buffer_t

    SmallVector<LLVM::LLVMType, 8> argTypes;
    SmallVector<Value, 8> argVals;

    auto llvmI8Ty = LLVM::LLVMType::getInt8Ty(llvmDialect);
    auto llvmI16Ty = LLVM::LLVMType::getInt16Ty(llvmDialect);
    auto llvmI32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);

    for (auto &operand : operands) {
      auto operTy = operand.getType();
      if (operTy.isa<IntegerType>() || operTy.isa<FloatType>()) {
        argTypes.push_back(typeConverter->convertType(operTy)
          .cast<LLVM::LLVMType>());
        argVals.push_back(operand);
      } else if (operTy.isa<MemRefType>()) {

        // memref is not yet converted to struct: this has something
        // to do with region signature conversion (I think that unlike
        // our first-step lowering (see comments in pass code), the second-step
        // (Standard->LLVM) lowering does trigger this region signature
        // conversion, so if the pattern would run in that latter context,
        // then the types would already be structs, not memrefs.

        auto structTy = typeConverter->convertType(operand.getType()).
          cast<LLVM::LLVMType>();
        auto ptrTy = structTy.getPointerTo();

        auto memRefType = operand.getType().cast<MemRefType>();
        auto memRefShape = memRefType.getShape();
        auto dims = memRefShape.size();

        Value one = rewriter.create<LLVM::ConstantOp>(loc,
            typeConverter->convertType(rewriter.getIndexType()),
            rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
        Value allocated =
            rewriter.create<LLVM::AllocaOp>(loc, ptrTy, one, /*alignment=*/0);

        Value dimsVal = rewriter.create<LLVM::ConstantOp>(loc,
            llvmI32Ty, rewriter.getI32IntegerAttr(dims));

        auto dimPtrTy = structTy.getStructElementType(kHalideBuffDim);
        auto dimElemTy = dimPtrTy.getPointerElementTy();
        auto dimArrTy = LLVM::LLVMType::getArrayTy(dimElemTy, dims);

        Value dimArrPtr = rewriter.create<LLVM::AllocaOp>(loc,
            dimArrTy.getPointerTo(), one, /*alignment=*/0);
        auto dimArrL = rewriter.create<LLVM::LoadOp>(loc, dimArrPtr);

        // TODO: HalideBuffDescriptor + StructBuilder

        auto descStructL = rewriter.create<LLVM::LoadOp>(loc, allocated);
        auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(llvmDialect);

        Value zero64 = rewriter.create<LLVM::ConstantOp>(loc,
            llvmI64Ty, rewriter.getI64IntegerAttr(0));
        Value zero32 = rewriter.create<LLVM::ConstantOp>(loc,
            llvmI32Ty, rewriter.getI64IntegerAttr(0));

        auto descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy,
            descStructL, zero64, rewriter.getI64ArrayAttr(kHalideBuffDevice));
        auto nullPtr = rewriter.create<LLVM::NullOp>(loc,
            llvmI8Ty.getPointerTo());
        descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy,
            descStruct, nullPtr,
            rewriter.getI64ArrayAttr(kHalideBuffDeviceInterface));

        // We need a real struct object, not the abstract memref type in order
        // to extract the runtime information (shape, etc). We could extract the
        // compiled-time info from the memref, but that does not include the
        // pointer to the allocated data array.
        auto memDescTy = typeConverter->convertType(operand.getType());

        static constexpr unsigned kAllocatedPtrPosInMemRefDescriptor = 0;
        auto dataPtr = rewriter.create<LLVM::ExtractValueOp>(loc,
            memDescTy, operand,
            rewriter.getI64ArrayAttr(kAllocatedPtrPosInMemRefDescriptor));

        // Element type in MemRef struct is strongly typed, but in
        // struct halide_buffer_t, it is not (generic uint8_t * + enum).
        // So, cast (in LLVM need two steps): [any]* -> int64_t -> int8_t*
        auto dataPtrI = rewriter.create<LLVM::PtrToIntOp>(loc,
            llvmI64Ty, dataPtr);
        auto dataPtrP = rewriter.create<LLVM::IntToPtrOp>(loc,
            llvmI8Ty.getPointerTo(), dataPtrI);

        descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy,
            descStruct, dataPtrP, rewriter.getI64ArrayAttr(kHalideBuffHost));

        descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy,
            descStruct, zero64, rewriter.getI64ArrayAttr(kHalideBuffFlags));

        uint8_t elemTypeCode;
        uint8_t elemTypeBits;
        uint16_t elemTypeLanes = 1; // default to scalar

        auto elemType = memRefType.getElementType();
        if (elemType.isa<FloatType>()) {
          elemTypeCode = kHalideTypeFloat;
          elemTypeBits = elemType.cast<FloatType>().getWidth();
        } else if (elemType.isa<IntegerType>()) {
          auto intTy = elemType.cast<IntegerType>();
          elemTypeCode = intTy.isSigned() ?  kHalideTypeInt : kHalideTypeUInt;
          elemTypeBits = intTy.getWidth();
        } else { // TODO: memref of vector element type possible?
          op->emitError("cannot lower memref of unsupported element type");
          return failure();
        }

        Value elemTypeCodeVal = rewriter.create<LLVM::ConstantOp>(loc,
            llvmI8Ty, rewriter.getI8IntegerAttr(elemTypeCode));
        descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy,
            descStruct, elemTypeCodeVal, rewriter.getI32ArrayAttr(
              {kHalideBuffType, kHalideBuffTypeCode}));
        Value elemTypeBitsVal = rewriter.create<LLVM::ConstantOp>(loc,
            llvmI8Ty, rewriter.getI8IntegerAttr(elemTypeBits));
        descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy,
            descStruct, elemTypeBitsVal, rewriter.getI32ArrayAttr(
              {kHalideBuffType, kHalideBuffTypeBits}));
        Value elemTypeLanesVal = rewriter.create<LLVM::ConstantOp>(loc,
            llvmI16Ty, rewriter.getI16IntegerAttr(elemTypeLanes));
        descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy,
            descStruct, elemTypeLanesVal, rewriter.getI32ArrayAttr(
              {kHalideBuffType, kHalideBuffTypeLanes}));

        int x_kHalideBuffDimensions = kHalideBuffDimensions;
        descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy,
            descStruct, dimsVal,
            rewriter.getI32ArrayAttr(x_kHalideBuffDimensions));

      // strides: note that buffers declared when the Halide pipeline is
      // defined need to have strides that match these memref strides (which
      // we set on the runtime Halide buffer descriptor below).
      int64_t offset; // TODO: figure out what this is and how to translate
      SmallVector<int64_t, 4> strides;
      bool strideSuccess = succeeded(getStridesAndOffset(memRefType,
            strides, offset));
      assert(strideSuccess && "no stride info in memref type");

      Value dimArr = dimArrL;

      for (int i = 0; i < dims; ++i) {
          // TODO: is this related to offset? (there is one offest, but a
          // min value per dimension... so not sure)
          Value minVal = zero32; // TODO
          dimArr = rewriter.create<LLVM::InsertValueOp>(loc, dimArrTy, dimArr,
              minVal, rewriter.getI32ArrayAttr(
                {i, kHalideBuffDimMin}));

          Value extentVal = rewriter.create<LLVM::ConstantOp>(loc,
            llvmI32Ty, rewriter.getI32IntegerAttr(memRefShape[i]));
          dimArr = rewriter.create<LLVM::InsertValueOp>(loc, dimArrTy, dimArr,
              extentVal, rewriter.getI32ArrayAttr(
                {i, kHalideBuffDimExtent}));

          Value strideVal = rewriter.create<LLVM::ConstantOp>(loc,
            llvmI32Ty, rewriter.getI32IntegerAttr(strides[i]));
          dimArr = rewriter.create<LLVM::InsertValueOp>(loc, dimArrTy, dimArr,
              strideVal, rewriter.getI32ArrayAttr(
                {i, kHalideBuffDimStride}));

          dimArr = rewriter.create<LLVM::InsertValueOp>(loc, dimArrTy, dimArr,
              zero32, rewriter.getI32ArrayAttr(
                {i, kHalideBuffDimFlags}));
        }

        rewriter.create<LLVM::StoreOp>(loc, dimArr, dimArrPtr);

        auto dimAddr = rewriter.create<LLVM::GEPOp>(loc, dimArrTy, dimArrPtr,
            ValueRange({zero32, zero32}));
        descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy, descStruct,
            dimAddr, rewriter.getI64ArrayAttr(kHalideBuffDim));

        dimArr = rewriter.create<LLVM::InsertValueOp>(loc, structTy, descStruct,
            nullPtr, rewriter.getI64ArrayAttr(kHalideBuffPadding));

        rewriter.create<LLVM::StoreOp>(loc, descStruct, allocated);

        argTypes.push_back(ptrTy);
        argVals.push_back(allocated);
      } else {
        // Should not happen, because argument type constraints in Ops.td
        op->emitError("unsupported operand type");
        return failure();
      }
    }

    // Get a symbol reference to the kernel function and declare it

    ArrayAttr variantsAttr = op->getAttrOfType<ArrayAttr>("variants");
    assert(variantsAttr); // TODO: verify

    StringAttr funcAttr = op->getAttrOfType<StringAttr>("func");
    assert(funcAttr); // verified
    std::string kernName = funcAttr.getValue().str();

    LLVM::LLVMType funcType = getKernFuncType(argTypes, llvmDialect);

	// Pass the map in a lazy way, effectively an iterator
	auto kernFuncMap = [&rewriter, &parentModule,
		 &kernName, &funcType, &variantsAttr](unsigned index) {
		unsigned variantId = variantsAttr[index].cast<IntegerAttr>().getInt();
		std::ostringstream funcName;
		funcName << kernName << "_v" << variantId;
		FlatSymbolRefAttr funcRef = getOrInsertKernFunc(rewriter, parentModule,
				funcName.str(), funcType);
		return std::pair<unsigned, FlatSymbolRefAttr>(variantId, funcRef);
	};
	JumpTable kernVariantsTable(rewriter, parentModule, typeConverter, loc,
			funcType, variantsAttr.size(), kernFuncMap);

    auto nodeId = callCRTGetNodeTypeId(rewriter, parentModule, llvmDialect,
			loc).getResult(0);

    Value funcPtr = kernVariantsTable.lookup(nodeId, loc);

    SmallVector<Value, 8> opArgVals{funcPtr};
    for (auto &arg : argVals) {
      opArgVals.push_back(arg);
    }

    IntegerAttr profileAttr = op->getAttrOfType<IntegerAttr>("profile");
    bool profile = profileAttr ? profileAttr.getInt() : false;
    std::cout << "profile: " << profile << std::endl;

    if (profile) {
      callCRTStopWatchStart(rewriter, parentModule, llvmDialect, loc);
    }

    auto newOp = rewriter.create<LLVM::CallOp>(op->getLoc(),
      LLVM::LLVMType::getVoidTy(llvmDialect), opArgVals,
      ArrayRef<NamedAttribute>{});

	if (profile) {
		auto swStopCall = callCRTStopWatchStop(rewriter, parentModule,
				llvmDialect, loc);
		Value elapsed = swStopCall.getResult(0);
		Value taskName = toy::allocString(rewriter, typeConverter,
                    llvmDialect, loc, op, "func");
		// TODO: need to call each variant
		Value variantId = createConstInt32(rewriter, typeConverter, loc, 1);

		callCRTLogMeasurement(rewriter, parentModule, llvmDialect, loc,
				{taskName, variantId, elapsed});
	}

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
  }

private:

  // The prototype of the kernel function, which is:
  //    `void (struct halide_buffer_t *, ...)`
  static LLVM::LLVMType getKernFuncType(ArrayRef<LLVM::LLVMType> args,
      LLVM::LLVMDialect *llvmDialect) {
      // Create a function declaration for the kernel, the signature is:
      //    `void (struct halide_buffer_t *, ...)`
      auto llvmVoidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
      auto llvmITy = LLVM::LLVMType::getInt64Ty(llvmDialect);
      auto llvmI8Ty = LLVM::LLVMType::getInt8Ty(llvmDialect);
      auto llvmDTy = LLVM::LLVMType::getDoubleTy(llvmDialect);

      return LLVM::LLVMType::getFunctionTy(llvmVoidTy, args,
          /*isVarArg*/ false);
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ToyToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct ToyToLLVMLoweringPass
    : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void runOnOperation() final;
};
} // end anonymous namespace

void ToyToLLVMLoweringPass::runOnOperation() {
  auto module = getOperation();

  // Have to lower in two steps:
  //   1. Lower HalideKernel all the way into LLVM dialect
  //   2. Lower everything else into LLVM dialect
  // Because, the HalideKernel lowering lowers memref types to LLVM structs,
  // but if we lower everything to LLVM dialect in one step, then all memref
  // types end up converted to structs, before HalideKernel conversion can
  // get to them.

  {
    // The first thing to define is the conversion target. This will define
    // the final target for this lowering. For this lowering, we are only
    // targeting the LLVM dialect.
    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<toy::ToyDialect>();
    target.addLegalOp<toy::PrintOp>();
    target.addLegalOp<toy::KernelOp>();
    target.addLegalOp<toy::PyKernelOp>();
    target.addLegalOp<LLVM::DialectCastOp>();

    ToyLLVMTypeConverter typeConverter(&getContext());

    OwningRewritePatternList patterns;
    patterns.insert<HalideKernelOpLowering>(typeConverter, &getContext());

    if (failed(applyPartialConversion(module, target, patterns))) {
      signalPassFailure();
    }
  }

  // During this lowering, we will also be lowering the MemRef types, that
  // are currently being operated on, to a representation in LLVM. To
  // perform this conversion we use a TypeConverter as part of the lowering.
  // This converter details how one type maps to another. This is necessary
  // now that we will be doing more complicated lowerings, involving loop
  // region arguments.

  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  LLVMTypeConverter typeConverter(&getContext());

  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process,
  // we have a combination of `toy`, `affine`, and `std` operations.
  // Luckily, there are already exists a set of patterns to transform
  // `affine` and `std` dialects. These patterns lowering in multiple
  // stages, relying on transitive lowerings. Transitive lowering, or
  // A->B->C lowering, is when multiple patterns must be applied to fully
  // transform an illegal operation into a set of legal ones.
  OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation to lower from the `toy` dialect, is the
  // PrintOp.
  patterns.insert<PrintOpLowering>(&getContext());
  patterns.insert<KernelOpLowering>(&getContext());
  patterns.insert<PyKernelOpLowering>(typeConverter, &getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(applyFullConversion(module, target, patterns)))
    signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::toy::createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}
