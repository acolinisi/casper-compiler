#ifndef CAC_BUILD_H
#define CAC_BUILD_H

namespace mlir {
  class MLIRContext;
  class OwningModuleRef;
}

namespace cac {
  class TaskGraph;
}

int buildMLIR(mlir::MLIRContext &context,
              mlir::OwningModuleRef &module);

// Metaprogram's main() is owned by the compiler, the app implements this
// "callback". TODO: let app own main().
int buildApp(cac::TaskGraph &tg);

int buildMLIRFromGraph(cac::TaskGraph &tg, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module);

#endif // CAC_BUILD_H
