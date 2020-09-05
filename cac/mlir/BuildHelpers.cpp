#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace toy {

// Allocate and fill a char[] array
// TODO: there has to be a better way to alloc and fill an array
// It needs to be alloced in .data, instead of on the stack.
  mlir::Value allocString(mlir::LLVM::LLVMDialect *llvmDialect,
    ConversionPatternRewriter &rewriter, TypeConverter *typeConverter,
    Location loc, StringRef value) {
  auto charTy = mlir::LLVM::LLVMType::getInt8Ty(llvmDialect);
  Value strLen = rewriter.create<mlir::LLVM::ConstantOp>(loc,
      typeConverter->convertType(rewriter.getIndexType()),
      rewriter.getIntegerAttr(rewriter.getIndexType(), value.size() + 1));
  // TODO: why shouldn't this be pointer to array type?
  Value charArr = rewriter.create<mlir::LLVM::AllocaOp>(loc,
      charTy.getPointerTo(), strLen, /*alignment=*/0);

  // Fill allocated array with string passed in 'value'
  for (int index = 0; index < value.size() + 1; ++index) {
    auto indexVal = rewriter.create<mlir::LLVM::ConstantOp>(loc,
        typeConverter->convertType(rewriter.getIndexType()),
        rewriter.getIntegerAttr(rewriter.getIndexType(), index));

    auto charAddr = rewriter.create<mlir::LLVM::GEPOp>(loc, charTy.getPointerTo(),
        charArr, ValueRange{indexVal});

    auto charVal = rewriter.create<mlir::LLVM::ConstantOp>(loc,
        typeConverter->convertType(rewriter.getIntegerType(8)),
      rewriter.getI8IntegerAttr(index < value.size() ? value[index] : 0));

    rewriter.create<mlir::LLVM::StoreOp>(loc, charVal, charAddr);
  }
  return charArr;
}

mlir::Value allocString(mlir::LLVM::LLVMDialect *llvmDialect,
    ConversionPatternRewriter &rewriter, TypeConverter *typeConverter,
    Location loc, Operation *op, StringRef attrName) {
  StringAttr attr = op->getAttrOfType<StringAttr>(attrName);
  assert(attr); // verified by .td
  return allocString(llvmDialect, rewriter, typeConverter,
      loc, attr.getValue());
}

} // namespace toy
} // namespace mlir
