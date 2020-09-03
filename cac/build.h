#pragma once

namespace mlir {
  class MLIRContext;
  class OwningModuleRef;
}

namespace cac {

class TaskGraph;
class Platform;
class KnowledgeBase;

int buildMLIRFromGraph(cac::TaskGraph &tg, cac::Platform &plat,
		mlir::MLIRContext &context, mlir::OwningModuleRef &module);

} // namespace cac
