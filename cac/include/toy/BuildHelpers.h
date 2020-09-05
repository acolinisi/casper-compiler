#pragma once

//#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
//#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Location.h"

namespace mlir {
    class ConversionPatternRewriter;
    class TypeConverter;

    namespace LLVM {
        class LLVMDialect;
    }
}

namespace mlir {
namespace toy {

Value allocString(mlir::LLVM::LLVMDialect *llvmDialect,
    ConversionPatternRewriter &rewriter, TypeConverter *typeConverter,
    Location loc, StringRef value);

mlir::Value allocString(mlir::LLVM::LLVMDialect *llvmDialect,
    ConversionPatternRewriter &rewriter, TypeConverter *typeConverter,
    Location loc, Operation *op, StringRef attrName);

} // namespace toy
} // namespace mlir
