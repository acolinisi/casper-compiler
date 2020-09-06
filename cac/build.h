#pragma once

#include <string>

namespace mlir {
  class MLIRContext;
  class OwningModuleRef;
}

namespace cac {

class TaskGraph;
class Platform;
class KnowledgeBase;

int buildMLIRFromGraph(mlir::OwningModuleRef &module, cac::TaskGraph &tg,
    cac::Platform &plat, mlir::MLIRContext &context,
    bool profilingHarness, const std::string &profilingMeasurementsFile);

} // namespace cac
