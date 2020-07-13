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

#if 0
#include "toy/Types.h"
#endif
#include "toy/Dialect.h"
#include "toy/Passes.h"

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

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {
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

      // Insert the printf function into the body of the parent module.
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, llvmFnType);
    }
    return SymbolRefAttr::get(name, context);
  }
};


#if 0


//===----------------------------------------------------------------------===//
// HalideBuffType
//===----------------------------------------------------------------------===//

/// Get or create a new HalideBuffType based on shape, element type, affine
/// map composition, and memory space.  Assumes the arguments define a
/// well-formed MemRef type.  Use getChecked to gracefully handle HalideBuffType
/// construction failures.
HalideBuffType HalideBuffType::get(ArrayRef<int64_t> shape, Type elementType,
                           ArrayRef<AffineMap> affineMapComposition,
                           unsigned memorySpace) {
  auto result = getImpl(shape, elementType, affineMapComposition, memorySpace,
                        /*location=*/llvm::None);
  assert(result && "Failed to construct instance of HalideBuffType.");
  return result;
}

/// Get or create a new HalideBuffType based on shape, element type, affine
/// map composition, and memory space declared at the given location.
/// If the location is unknown, the last argument should be an instance of
/// UnknownLoc.  If the HalideBuffType defined by the arguments would be
/// ill-formed, emits errors (to the handler registered with the context or to
/// the error stream) and returns nullptr.
HalideBuffType HalideBuffType::getChecked(ArrayRef<int64_t> shape, Type elementType,
                                  ArrayRef<AffineMap> affineMapComposition,
                                  unsigned memorySpace, Location location) {
  return getImpl(shape, elementType, affineMapComposition, memorySpace,
                 location);
}

/// Get or create a new HalideBuffType defined by the arguments.  If the resulting
/// type would be ill-formed, return nullptr.  If the location is provided,
/// emit detailed error messages.  To emit errors when the location is unknown,
/// pass in an instance of UnknownLoc.
HalideBuffType HalideBuffType::getImpl(ArrayRef<int64_t> shape, Type elementType,
                               ArrayRef<AffineMap> affineMapComposition,
                               unsigned memorySpace,
                               Optional<Location> location) {
  auto *context = elementType.getContext();

  // Check that memref is formed from allowed types.
  if (!elementType.isIntOrFloat() && !elementType.isa<VectorType>() &&
      !elementType.isa<ComplexType>())
    return emitOptionalError(location, "invalid memref element type"),
           HalideBuffType();

  for (int64_t s : shape) {
    // Negative sizes are not allowed except for `-1` that means dynamic size.
    if (s < -1)
      return emitOptionalError(location, "invalid memref size"), HalideBuffType();
  }

  // Check that the structure of the composition is valid, i.e. that each
  // subsequent affine map has as many inputs as the previous map has results.
  // Take the dimensionality of the MemRef for the first map.
  auto dim = shape.size();
  unsigned i = 0;
  for (const auto &affineMap : affineMapComposition) {
    if (affineMap.getNumDims() != dim) {
      if (location)
        emitError(*location)
            << "memref affine map dimension mismatch between "
            << (i == 0 ? Twine("memref rank") : "affine map " + Twine(i))
            << " and affine map" << i + 1 << ": " << dim
            << " != " << affineMap.getNumDims();
      return nullptr;
    }

    dim = affineMap.getNumResults();
    ++i;
  }

  // Drop identity maps from the composition.
  // This may lead to the composition becoming empty, which is interpreted as an
  // implicit identity.
  SmallVector<AffineMap, 2> cleanedAffineMapComposition;
  for (const auto &map : affineMapComposition) {
    if (map.isIdentity())
      continue;
    cleanedAffineMapComposition.push_back(map);
  }

  return Base::get(context, CacTypes::HalideBuffType, shape, elementType,
                   cleanedAffineMapComposition, memorySpace);
}

ArrayRef<int64_t> HalideBuffType::getShape() const { return getImpl()->getShape(); }

ArrayRef<AffineMap> HalideBuffType::getAffineMaps() const {
  return getImpl()->getAffineMaps();
}

unsigned HalideBuffType::getMemorySpace() const { return getImpl()->memorySpace; }

/// Helper class to produce LLVM dialect operations extracting or inserting /
//elements of a struct halide_buffer_t. Wraps a Value pointing to the
//descriptor.
/// The Value may be null, in which case none of the operations are valid.
class HalideBuffDescriptor : public StructBuilder {
public:
  /// Construct a helper for the given descriptor value.
  explicit HalideBuffDescriptor(Value descriptor);
  /// Builds IR creating an `undef` value of the descriptor type.
  static HalideBuffDescriptor undef(OpBuilder &builder, Location loc,
                                Type descriptorType);
#if 0
  /// Builds IR creating a HalideBuff descriptor that represents `type` and
  /// populates it with static shape and stride information extracted from the
  /// type.
  static HalideBuffDescriptor fromStaticShape(OpBuilder &builder, Location loc,
                                          LLVMTypeConverter &typeConverter,
                                          HalideBuffType type, Value memory);
#endif

  /// Builds IR extracting the allocated pointer from the descriptor.
  Value allocatedPtr(OpBuilder &builder, Location loc);
  /// Builds IR inserting the allocated pointer into the descriptor.
  void setAllocatedPtr(OpBuilder &builder, Location loc, Value ptr);

#if 0
  /// Builds IR extracting the aligned pointer from the descriptor.
  Value alignedPtr(OpBuilder &builder, Location loc);

  /// Builds IR inserting the aligned pointer into the descriptor.
  void setAlignedPtr(OpBuilder &builder, Location loc, Value ptr);

  /// Builds IR extracting the offset from the descriptor.
  Value offset(OpBuilder &builder, Location loc);

  /// Builds IR inserting the offset into the descriptor.
  void setOffset(OpBuilder &builder, Location loc, Value offset);
  void setConstantOffset(OpBuilder &builder, Location loc, uint64_t offset);
#endif

  /// Builds IR extracting the pos-th size from the descriptor.
  Value size(OpBuilder &builder, Location loc, unsigned pos);
  Value size(OpBuilder &builder, Location loc, Value pos, int64_t rank);

  /// Builds IR inserting the pos-th size into the descriptor
  void setSize(OpBuilder &builder, Location loc, unsigned pos, Value size);
  void setConstantSize(OpBuilder &builder, Location loc, unsigned pos,
                       uint64_t size);

#if 0
  /// Builds IR extracting the pos-th size from the descriptor.
  Value stride(OpBuilder &builder, Location loc, unsigned pos);

  /// Builds IR inserting the pos-th stride into the descriptor
  void setStride(OpBuilder &builder, Location loc, unsigned pos, Value stride);
  void setConstantStride(OpBuilder &builder, Location loc, unsigned pos,
                         uint64_t stride);
#endif

  /// Returns the (LLVM) type this descriptor points to.
  LLVM::LLVMType getElementType();

  /// Builds IR populating a HalideBuff descriptor structure from a list of
  /// individual values composing that descriptor, in the following order:
  /// - allocated pointer;
  /// - aligned pointer;
  /// - offset;
  /// - <rank> sizes;
  /// - <rank> shapes;
  /// where <rank> is the HalideBuff rank as provided in `type`.
  static Value pack(OpBuilder &builder, Location loc,
                    LLVMTypeConverter &converter, HalideBuffType type,
                    ValueRange values);

#if 0
  /// Builds IR extracting individual elements of a HalideBuff descriptor
  //structure / and returning them as `results` list.
  static void unpack(OpBuilder &builder, Location loc, Value packed,
                     HalideBuffType type, SmallVectorImpl<Value> &results);
#endif

  /// Returns the number of non-aggregate values that would be produced by
  /// `unpack`.
  static unsigned getNumUnpackedValues(HalideBuffType type);

private:
  // Cached index type.
  Type indexType;
};


/*============================================================================*/
/* HalideBuffDescriptor implementation                                            */
/*============================================================================*/

/// Construct a helper for the given descriptor value.
HalideBuffDescriptor::HalideBuffDescriptor(Value descriptor)
    : StructBuilder(descriptor) {
  assert(value != nullptr && "value cannot be null");
  indexType = value.getType().cast<LLVM::LLVMType>().getStructElementType(
      kOffsetPosInHalideBuffDescriptor);
}

/// Builds IR creating an `undef` value of the descriptor type.
HalideBuffDescriptor HalideBuffDescriptor::undef(OpBuilder &builder, Location loc,
                                         Type descriptorType) {

  Value descriptor =
      builder.create<LLVM::UndefOp>(loc, descriptorType.cast<LLVM::LLVMType>());
  return HalideBuffDescriptor(descriptor);
}

#if 0
/// Builds IR creating a MemRef descriptor that represents `type` and
/// populates it with static shape and stride information extracted from the
/// type.
HalideBuffDescriptor
HalideBuffDescriptor::fromStaticShape(OpBuilder &builder, Location loc,
                                  LLVMTypeConverter &typeConverter,
                                  MemRefType type, Value memory) {
  assert(type.hasStaticShape() && "unexpected dynamic shape");

  // Extract all strides and offsets and verify they are static.
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto result = getStridesAndOffset(type, strides, offset);
  (void)result;
  assert(succeeded(result) && "unexpected failure in stride computation");
  assert(offset != MemRefType::getDynamicStrideOrOffset() &&
         "expected static offset");
  assert(!llvm::is_contained(strides, MemRefType::getDynamicStrideOrOffset()) &&
         "expected static strides");

  auto convertedType = typeConverter.convertType(type);
  assert(convertedType && "unexpected failure in memref type conversion");

  auto descr = HalideBuffDescriptor::undef(builder, loc, convertedType);
  descr.setAllocatedPtr(builder, loc, memory);
  descr.setAlignedPtr(builder, loc, memory);
  descr.setConstantOffset(builder, loc, offset);

  // Fill in sizes and strides
  for (unsigned i = 0, e = type.getRank(); i != e; ++i) {
    descr.setConstantSize(builder, loc, i, type.getDimSize(i));
    descr.setConstantStride(builder, loc, i, strides[i]);
  }
  return descr;
}
#endif

/// Builds IR extracting the allocated pointer from the descriptor.
Value HalideBuffDescriptor::allocatedPtr(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kAllocatedPtrPosInHalideBuffDescriptor);
}

/// Builds IR inserting the allocated pointer into the descriptor.
void HalideBuffDescriptor::setAllocatedPtr(OpBuilder &builder, Location loc,
                                       Value ptr) {
  setPtr(builder, loc, kAllocatedPtrPosInHalideBuffDescriptor, ptr);
}

#if 0
/// Builds IR extracting the aligned pointer from the descriptor.
Value HalideBuffDescriptor::alignedPtr(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kAlignedPtrPosInHalideBuffDescriptor);
}

/// Builds IR inserting the aligned pointer into the descriptor.
void HalideBuffDescriptor::setAlignedPtr(OpBuilder &builder, Location loc,
                                     Value ptr) {
  setPtr(builder, loc, kAlignedPtrPosInHalideBuffDescriptor, ptr);
}

// Creates a constant Op producing a value of `resultType` from an index-typed
// integer attribute.
static Value createIndexAttrConstant(OpBuilder &builder, Location loc,
                                     Type resultType, int64_t value) {
  return builder.create<LLVM::ConstantOp>(
      loc, resultType, builder.getIntegerAttr(builder.getIndexType(), value));
}

/// Builds IR extracting the offset from the descriptor.
Value HalideBuffDescriptor::offset(OpBuilder &builder, Location loc) {
  return builder.create<LLVM::ExtractValueOp>(
      loc, indexType, value,
      builder.getI64ArrayAttr(kOffsetPosInHalideBuffDescriptor));
}

/// Builds IR inserting the offset into the descriptor.
void HalideBuffDescriptor::setOffset(OpBuilder &builder, Location loc,
                                 Value offset) {
  value = builder.create<LLVM::InsertValueOp>(
      loc, structType, value, offset,
      builder.getI64ArrayAttr(kOffsetPosInHalideBuffDescriptor));
}

/// Builds IR inserting the offset into the descriptor.
void HalideBuffDescriptor::setConstantOffset(OpBuilder &builder, Location loc,
                                         uint64_t offset) {
  setOffset(builder, loc,
            createIndexAttrConstant(builder, loc, indexType, offset));
}
#endif

/// Builds IR extracting the pos-th size from the descriptor.
Value HalideBuffDescriptor::size(OpBuilder &builder, Location loc, unsigned pos) {
  return builder.create<LLVM::ExtractValueOp>(
      loc, indexType, value,
      builder.getI64ArrayAttr({kSizePosInHalideBuffDescriptor, pos}));
}

Value HalideBuffDescriptor::size(OpBuilder &builder, Location loc, Value pos,
                             int64_t rank) {
  auto indexTy = indexType.cast<LLVM::LLVMType>();
  auto indexPtrTy = indexTy.getPointerTo();
  auto arrayTy = LLVM::LLVMType::getArrayTy(indexTy, rank);
  auto arrayPtrTy = arrayTy.getPointerTo();

  // Copy size values to stack-allocated memory.
  auto zero = createIndexAttrConstant(builder, loc, indexType, 0);
  auto one = createIndexAttrConstant(builder, loc, indexType, 1);
  auto sizes = builder.create<LLVM::ExtractValueOp>(
      loc, arrayTy, value,
      builder.getI64ArrayAttr({kSizePosInHalideBuffDescriptor}));
  auto sizesPtr =
      builder.create<LLVM::AllocaOp>(loc, arrayPtrTy, one, /*alignment=*/0);
  builder.create<LLVM::StoreOp>(loc, sizes, sizesPtr);

  // Load an return size value of interest.
  auto resultPtr = builder.create<LLVM::GEPOp>(loc, indexPtrTy, sizesPtr,
                                               ValueRange({zero, pos}));
  return builder.create<LLVM::LoadOp>(loc, resultPtr);
}

/// Builds IR inserting the pos-th size into the descriptor
void HalideBuffDescriptor::setSize(OpBuilder &builder, Location loc, unsigned pos,
                               Value size) {
  value = builder.create<LLVM::InsertValueOp>(
      loc, structType, value, size,
      builder.getI64ArrayAttr({kSizePosInHalideBuffDescriptor, pos}));
}

void HalideBuffDescriptor::setConstantSize(OpBuilder &builder, Location loc,
                                       unsigned pos, uint64_t size) {
  setSize(builder, loc, pos,
          createIndexAttrConstant(builder, loc, indexType, size));
}

#if 0
/// Builds IR extracting the pos-th stride from the descriptor.
Value HalideBuffDescriptor::stride(OpBuilder &builder, Location loc, unsigned pos) {
  return builder.create<LLVM::ExtractValueOp>(
      loc, indexType, value,
      builder.getI64ArrayAttr({kStridePosInHalideBuffDescriptor, pos}));
}

/// Builds IR inserting the pos-th stride into the descriptor
void HalideBuffDescriptor::setStride(OpBuilder &builder, Location loc, unsigned pos,
                                 Value stride) {
  value = builder.create<LLVM::InsertValueOp>(
      loc, structType, value, stride,
      builder.getI64ArrayAttr({kStridePosInHalideBuffDescriptor, pos}));
}

void HalideBuffDescriptor::setConstantStride(OpBuilder &builder, Location loc,
                                         unsigned pos, uint64_t stride) {
  setStride(builder, loc, pos,
            createIndexAttrConstant(builder, loc, indexType, stride));
}
#endif

LLVM::LLVMType HalideBuffDescriptor::getElementType() {
  return value.getType().cast<LLVM::LLVMType>().getStructElementType(
      kAlignedPtrPosInHalideBuffDescriptor);
}

/// Creates a MemRef descriptor structure from a list of individual values
/// composing that descriptor, in the following order:
/// - allocated pointer;
/// - aligned pointer;
/// - offset;
/// - <rank> sizes;
/// - <rank> shapes;
/// where <rank> is the MemRef rank as provided in `type`.
Value HalideBuffDescriptor::pack(OpBuilder &builder, Location loc,
                             LLVMTypeConverter &converter, MemRefType type,
                             ValueRange values) {
  Type llvmType = converter.convertType(type);
  auto d = HalideBuffDescriptor::undef(builder, loc, llvmType);

  d.setAllocatedPtr(builder, loc, values[kAllocatedPtrPosInHalideBuffDescriptor]);
  d.setAlignedPtr(builder, loc, values[kAlignedPtrPosInHalideBuffDescriptor]);
  d.setOffset(builder, loc, values[kOffsetPosInHalideBuffDescriptor]);

  int64_t rank = type.getRank();
  for (unsigned i = 0; i < rank; ++i) {
    d.setSize(builder, loc, i, values[kSizePosInHalideBuffDescriptor + i]);
    d.setStride(builder, loc, i, values[kSizePosInHalideBuffDescriptor + rank + i]);
  }

  return d;
}

#if 0
/// Builds IR extracting individual elements of a MemRef descriptor structure
/// and returning them as `results` list.
void HalideBuffDescriptor::unpack(OpBuilder &builder, Location loc, Value packed,
                              MemRefType type,
                              SmallVectorImpl<Value> &results) {
  int64_t rank = type.getRank();
  results.reserve(results.size() + getNumUnpackedValues(type));

  HalideBuffDescriptor d(packed);
  results.push_back(d.allocatedPtr(builder, loc));
  results.push_back(d.alignedPtr(builder, loc));
  results.push_back(d.offset(builder, loc));
  for (int64_t i = 0; i < rank; ++i)
    results.push_back(d.size(builder, loc, i));
  for (int64_t i = 0; i < rank; ++i)
    results.push_back(d.stride(builder, loc, i));
}
#endif

/// Returns the number of non-aggregate values that would be produced by
/// `unpack`.
unsigned HalideBuffDescriptor::getNumUnpackedValues(MemRefType type) {
  // Two pointers, offset, <rank> sizes, <rank> shapes.
  return 3 + 2 * type.getRank();
}

#endif // HalideBufType (everything)

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

// TODO: change to construct type of sturct halide_buffer_t
Type ToyLLVMTypeConverter::convertMemRefTypeToStructHalideBuf(MemRefType type) {

#if 0
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  bool strideSuccess = succeeded(getStridesAndOffset(type, strides, offset));
  assert(strideSuccess &&
         "Non-strided layout maps must have been normalized away");
  (void)strideSuccess;
#endif


  LLVM::LLVMType elementType = unwrap(convertType(type.getElementType()));
  if (!elementType)
    return {};

  LLVM::LLVMDialect *llvmDialect = &elementType.getDialect();

  auto memSpace = type.getMemorySpace();
  auto int64Ty = LLVM::LLVMType::getInt64Ty(llvmDialect);
  auto int32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto int16Ty = LLVM::LLVMType::getInt16Ty(llvmDialect);
  auto int8Ty = LLVM::LLVMType::getInt8Ty(llvmDialect);
  //auto voidPtrTy = LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo(memSpace);
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
public:
  explicit HalideKernelOpLowering(TypeConverter &typeConverterArg,
      TypeConverter &llvmTypeConverter,
      MLIRContext *context)
      : ConversionPattern(toy::HalideKernelOp::getOperationName(), 1, context), typeConverterM(typeConverterArg), llvmTypeConverter(llvmTypeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    toy::HalideKernelOp opx = llvm::dyn_cast<toy::HalideKernelOp>(op);
    printf("LOWERING %p!!!\n", opx);
#if 0
    auto halideBufType = (*op->operand_type_begin()).cast<F64HalideBuff>();
#else
#endif

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

#if 0
    // TODO: descTy will be a fixed type constructed inside getOrInsertKernFunc
    auto descTy = operands[0].getType().cast<LLVM::LLVMType>().getPointerTo();
    auto kernRef = getOrInsertKernFunc(rewriter, parentModule,
        func, descTy, op->getNumOperands(), llvmDialect);
    auto kernOp = cast<toy::HalideKernelOp>(op);

    auto memRefArgs = kernOp.input();
#endif

    // MemRefType -> HalideBuffType -> struct pointer

    // TODO: convert memrefArgs to struct halide_buffer_t (use StructBuilder)
    // TODO: CallOp->CallOp, with Type converter for MemRefType->HalideBuffType
    // rewriter.create<CallOp>(loc, kernRef, ArrayRef<Type>());
  
    // TODO: each operand is an LLVM struct, but it's the struct memref_t,
    // not yet struct halide_buffer_t. MemRefType has been converted to
    // struct type by convertMemRefType conversion in std->llvm type
    // converter.

    // TODO: this is the same for all memref args
    FlatSymbolRefAttr kernRef;

    SmallVector<Value, 8> args;

    auto llvmI8Ty = LLVM::LLVMType::getInt8Ty(llvmDialect);
    auto llvmI16Ty = LLVM::LLVMType::getInt16Ty(llvmDialect);
    auto llvmI32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);

    // TODO: offset arg: this is a non-dat arg of the Halide pipeline
    //       pass via attributes?
#if 0
    Value zero = rewriter.create<LLVM::ConstantOp>(loc,
        llvmI8Ty, rewriter.getI8IntegerAttr(0));
    args.push_back(zero);
#else
    Value one = rewriter.create<LLVM::ConstantOp>(loc,
        llvmI8Ty, rewriter.getI8IntegerAttr(1));
    args.push_back(one);
#endif

#if 0
    for (auto &operand : operands) {
#endif

    for (const auto &pair : llvm::zip(op->getOperands(), operands)) {
      auto origOperand = std::get<0>(pair);
      auto operand = std::get<1>(pair);
      printf("LOWERING OPERAND!!!\n");

#if 0 // TODO: operants have already been converted
      auto ptrTy = operand.getType().cast<LLVM::LLVMType>().getPointerTo();
#else
      auto structTy = typeConverterM.convertType(operand.getType()).
        cast<LLVM::LLVMType>();
      auto ptrTy = structTy.getPointerTo();
#endif
      // fields from struct halide_buffer_t in HalideRuntime.h
      int kHalideBuffDevice = 0;
      int kHalideBuffDeviceInterface = 1;
      int kHalideBuffHost = 2;
      int kHalideBuffFlags = 3;
      int kHalideBuffType = 4;
      int kHalideBuffDimensions = 5;
      int kHalideBuffDim = 6;
      int kHalideBuffPadding = 7;

      int kHalideBuffTypeCode = 0;
      int kHalideBuffTypeBits = 1;
      int kHalideBuffTypeLanes = 2;

      int kHalideBuffDimMin = 0;
      int kHalideBuffDimExtent = 1;
      int kHalideBuffDimStride = 2;
      int kHalideBuffDimFlags = 3;

      auto memRefType = origOperand.getType().cast<MemRefType>();
      auto memRefShape = memRefType.getShape();
      auto dims = memRefShape.size();

      kernRef = getOrInsertKernFunc(rewriter, parentModule,
          func, ptrTy, op->getNumOperands(), llvmDialect);

      Value one = rewriter.create<LLVM::ConstantOp>(loc,
          typeConverterM.convertType(rewriter.getIndexType()),
          rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
      Value allocated =
          rewriter.create<LLVM::AllocaOp>(loc, ptrTy, one, /*alignment=*/0);

      Value dimsVal = rewriter.create<LLVM::ConstantOp>(loc,
          llvmI32Ty, rewriter.getI32IntegerAttr(dims));

#if 0
      auto dimArrTy = LLVM::LLVMType::getArrayTy(
          dimPtrTy.getPointerElementTy(), dims);
#else
      auto dimPtrTy = structTy.getStructElementType(kHalideBuffDim);
      auto dimElemTy = dimPtrTy.getPointerElementTy();
      auto dimArrTy = LLVM::LLVMType::getArrayTy(dimElemTy, dims);
#endif

#if 0
      Value dimPtr = rewriter.create<LLVM::AllocaOp>(loc,
          dimPtrTy, dimsVal, /*alignment=*/0);
#else
      Value dimArrPtr = rewriter.create<LLVM::AllocaOp>(loc,
          dimArrTy.getPointerTo(), one, /*alignment=*/0);
      auto dimArrL = rewriter.create<LLVM::LoadOp>(loc, dimArrPtr);
#endif

      // TODO: HalideBuffDescriptor + StructBuilder

      auto descStructL = rewriter.create<LLVM::LoadOp>(loc, allocated);
      auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(llvmDialect);

      Value zero64 = rewriter.create<LLVM::ConstantOp>(loc,
          llvmI64Ty, rewriter.getI64IntegerAttr(0));
      Value zero32 = rewriter.create<LLVM::ConstantOp>(loc,
          llvmI32Ty, rewriter.getI64IntegerAttr(0));

      auto descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy, descStructL,
          zero64, rewriter.getI64ArrayAttr(kHalideBuffDevice));
      auto nullPtr = rewriter.create<LLVM::NullOp>(loc,
          llvmI8Ty.getPointerTo());
      descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy, descStruct,
          nullPtr, rewriter.getI64ArrayAttr(kHalideBuffDeviceInterface));

      auto memDescTy = llvmTypeConverter.convertType(operand.getType());

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

      descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy, descStruct,
          dataPtrP, rewriter.getI64ArrayAttr(kHalideBuffHost));

      descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy, descStruct,
          zero64, rewriter.getI64ArrayAttr(kHalideBuffFlags));

      uint8_t elemTypeCode = 2; // float TODO: take from MemRef type
      uint8_t elemTypeBits = 64;
      uint16_t elemTypeLanes = 1;
      Value elemTypeCodeVal = rewriter.create<LLVM::ConstantOp>(loc,
          llvmI8Ty, rewriter.getI8IntegerAttr(elemTypeCode));
      descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy, descStruct,
          elemTypeCodeVal, rewriter.getI32ArrayAttr(
            {kHalideBuffType, kHalideBuffTypeCode}));
      Value elemTypeBitsVal = rewriter.create<LLVM::ConstantOp>(loc,
          llvmI8Ty, rewriter.getI8IntegerAttr(elemTypeBits));
      descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy, descStruct,
          elemTypeBitsVal, rewriter.getI32ArrayAttr(
            {kHalideBuffType, kHalideBuffTypeBits}));
      Value elemTypeLanesVal = rewriter.create<LLVM::ConstantOp>(loc,
          llvmI16Ty, rewriter.getI16IntegerAttr(elemTypeLanes));
      descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy, descStruct,
          elemTypeLanesVal, rewriter.getI32ArrayAttr(
            {kHalideBuffType, kHalideBuffTypeLanes}));

      descStruct = rewriter.create<LLVM::InsertValueOp>(loc, structTy, descStruct,
          dimsVal, rewriter.getI32ArrayAttr(kHalideBuffDimensions));

    int64_t offset; // TODO: figure out what this is and how to translate
    SmallVector<int64_t, 4> strides;
    bool strideSuccess = succeeded(getStridesAndOffset(memRefType,
          strides, offset));
    assert(strideSuccess && "no stride info in memref type");

    Value dimArr = dimArrL;

    for (int i = 0; i < dims; ++i) {
        Value minVal = rewriter.create<LLVM::ConstantOp>(loc,
          llvmI32Ty, rewriter.getI32IntegerAttr(0));
        dimArr = rewriter.create<LLVM::InsertValueOp>(loc, dimArrTy, dimArr,
            minVal, rewriter.getI32ArrayAttr(
              {i, kHalideBuffDimMin}));

        Value extentVal = rewriter.create<LLVM::ConstantOp>(loc,
          llvmI32Ty, rewriter.getI32IntegerAttr(memRefShape[i]));
        dimArr = rewriter.create<LLVM::InsertValueOp>(loc, dimArrTy, dimArr,
            extentVal, rewriter.getI32ArrayAttr(
              {i, kHalideBuffDimExtent}));

#if 0 // TODO
        Value strideVal = rewriter.create<LLVM::ConstantOp>(loc,
          llvmI32Ty, rewriter.getI32IntegerAttr(strides[i]));
#else
        int stride = (i == 0) ? 1 : 3; // TODO: hack
        Value strideVal = rewriter.create<LLVM::ConstantOp>(loc,
          llvmI32Ty, rewriter.getI32IntegerAttr(stride));
#endif
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

      args.push_back(allocated);
    }


    auto newOp = rewriter.create<LLVM::CallOp>(op->getLoc(),
      LLVM::LLVMType::getVoidTy(llvmDialect), args,
      ArrayRef<NamedAttribute>{NamedAttribute(Identifier::get(StringRef("callee", 6), parentModule.getContext()), kernRef)});

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    printf("LOWERING ERASE!!!\n");
    return success();
  }

private:
  /// Return a symbol reference to the kernel function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertKernFunc(PatternRewriter &rewriter,
                                             ModuleOp module,
                                             StringRef name,
                                             LLVM::LLVMType descTy,
                                             int numOpers,
                                             LLVM::LLVMDialect *llvmDialect) {
    auto *context = module.getContext();
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {

      // Create a function declaration for the kernel, the signature is:
      //    `void (struct halide_buffer_t *, ...)`
      auto llvmVoidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
      auto llvmITy = LLVM::LLVMType::getInt64Ty(llvmDialect);
      auto llvmI8Ty = LLVM::LLVMType::getInt8Ty(llvmDialect);
      auto llvmDTy = LLVM::LLVMType::getDoubleTy(llvmDialect);

      std::vector<LLVM::LLVMType> opers;
      opers.push_back(llvmI8Ty); // offset
      for (int i = 0; i < numOpers; ++i) {
#if 0
        opers.push_back(llvmDTy.getPointerTo()); /* buffer */
        opers.push_back(llvmDTy.getPointerTo()); /* start of aligned data */
        opers.push_back(llvmITy); /* offset into buffer */
        opers.push_back(llvmITy); /* size per dim */
        opers.push_back(llvmITy);
        opers.push_back(llvmITy); /* stride per dim */
        opers.push_back(llvmITy);
#else
        opers.push_back(descTy);
#endif
      }

      // TODO: unranked memref?
      auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmVoidTy,
        opers, /*isVarArg*/ false);

      // Insert the printf function into the body of the parent module.
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, llvmFnType);
    }
    return SymbolRefAttr::get(name, context);
  }

  TypeConverter &typeConverterM;
  TypeConverter &llvmTypeConverter;
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
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.

  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. To perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  LLVMTypeConverter typeConverter(&getContext());
#if 1
  {
  LLVMConversionTarget target(getContext());
  target.addIllegalDialect<toy::ToyDialect>();
  //target.addLegalOp<toy::HalideKernelOp>(); // TODO: constrain arg types
  target.addLegalOp<toy::PrintOp>();

  OwningRewritePatternList patterns;
  ToyLLVMTypeConverter toyTypeConverter(&getContext());


#if 0
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());


  // The only remaining operation to lower from the `toy` dialect, is the
  // PrintOp.
  patterns.insert<PrintOpLowering>(&getContext());
  patterns.insert<KernelOpLowering>(&getContext());
#endif


  patterns.insert<HalideKernelOpLowering>(toyTypeConverter, typeConverter,
      &getContext());

  //printf("LLVM PASS!!\n");
  auto module = getOperation();
  if (failed(applyPartialConversion(module, target, patterns))) {
    //printf("FAILED!!\n");
    //module.dump();
    signalPassFailure();
  }
  //module.dump();
#if 0
  return;
#endif
  }
#endif

#if 1
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
#endif



  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());


  // The only remaining operation to lower from the `toy` dialect, is the
  // PrintOp.
  patterns.insert<PrintOpLowering>(&getContext());
  patterns.insert<KernelOpLowering>(&getContext());

#if 0
  ToyLLVMTypeConverter toyTypeConverter(&getContext());
  patterns.insert<HalideKernelOpLowering>(toyTypeConverter, &getContext());
#endif

#if 1
  populateStdToLLVMConversionPatterns(typeConverter, patterns);
#endif

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
#if 0
  if (failed(applyFullConversion(module, target, patterns)))
    signalPassFailure();
#else
  if (failed(applyPartialConversion(module, target, patterns))) {
    //module.dump();
    signalPassFailure();
  }
#endif

  //module.dump();
  //printf("DONE!!!!!\n");
#if 0
  OwningRewritePatternList patterns2;
  populateAffineToStdConversionPatterns(patterns2, &getContext());
  populateLoopToStdConversionPatterns(patterns2, &getContext());
  populateStdToLLVMConversionPatterns(typeConverter, patterns2);
  if (failed(applyPartialConversion(module, target2, patterns2))) {
    module.dump();
    signalPassFailure();
  }
#endif
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::toy::createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}
