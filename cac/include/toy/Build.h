#ifndef CAC_BUILD_H
#define CAC_BUILD_H

namespace mlir {
  class MLIRContext;
  class OwningModuleRef;
}

int buildMLIR(mlir::MLIRContext &context,
              mlir::OwningModuleRef &module);

#endif // CAC_BUILD_H
