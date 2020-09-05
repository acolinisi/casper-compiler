#pragma once

#include "mlir/IR/Location.h"

namespace mlir {
    class OpBuilder;
    class TypeConverter;

    namespace LLVM {
        class LLVMDialect;
    }
}

namespace mlir {
namespace toy {

Value allocString(OpBuilder &builder, TypeConverter *typeConverter,
    mlir::LLVM::LLVMDialect *llvmDialect,
    Location loc, StringRef value);

Value allocString(OpBuilder &builder, TypeConverter *typeConverter,
    mlir::LLVM::LLVMDialect *llvmDialect,
    Location loc, Operation *op, StringRef attrName);

} // namespace toy
} // namespace mlir
