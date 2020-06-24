#ifndef CAC_BUILD_H
#define CAC_BUILD_H

namespace mlir {
  class MLIRContext;
  class OwningModuleRef;
}

namespace cac {
  class TaskGraph;
}

int buildMLIRFromGraph(cac::TaskGraph &tg, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module);

#endif // CAC_BUILD_H
