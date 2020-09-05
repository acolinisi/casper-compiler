#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace toy {

// Allocate and fill a char[] array
// TODO: there has to be a better way to alloc and fill an array
// It needs to be alloced in .data, instead of on the stack.
Value allocString(OpBuilder &builder, mlir::LLVM::LLVMDialect *llvmDialect,
    Location loc, StringRef value) {
  auto charTy = mlir::LLVM::LLVMType::getInt8Ty(llvmDialect);
  auto idxTy = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  Value strLen = builder.create<mlir::LLVM::ConstantOp>(loc, idxTy,
      builder.getIntegerAttr(builder.getIndexType(), value.size() + 1));
  // TODO: why shouldn't this be pointer to array type?
  Value charArr = builder.create<mlir::LLVM::AllocaOp>(loc,
      charTy.getPointerTo(), strLen, /*alignment=*/0);

  // Fill allocated array with string passed in 'value'
  for (int index = 0; index < value.size() + 1; ++index) {
    auto indexVal = builder.create<mlir::LLVM::ConstantOp>(loc, idxTy,
        builder.getIntegerAttr(builder.getIndexType(), index));

    auto charAddr = builder.create<mlir::LLVM::GEPOp>(loc, charTy.getPointerTo(),
        charArr, ValueRange{indexVal});
    auto charVal = builder.create<mlir::LLVM::ConstantOp>(loc, charTy,
      builder.getI8IntegerAttr(index < value.size() ? value[index] : 0));

    builder.create<mlir::LLVM::StoreOp>(loc, charVal, charAddr);
  }
  return charArr;
}

Value allocString(OpBuilder &builder, mlir::LLVM::LLVMDialect *llvmDialect,
    Location loc, Operation *op, StringRef attrName) {
  StringAttr attr = op->getAttrOfType<StringAttr>(attrName);
  assert(attr); // TODO: be nicer
  return allocString(builder, llvmDialect, loc, attr.getValue());
}

} // namespace toy
} // namespace mlir
