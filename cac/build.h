#pragma once

#include <string>
#include <vector>

namespace mlir {
  class MLIRContext;
  class OwningModuleRef;
}

namespace cac {

class TaskGraph;
class Platform;
class KnowledgeBase;

int buildMLIRFromGraph(mlir::OwningModuleRef &module, cac::TaskGraph &tg,
    std::vector<unsigned> variantIds, mlir::MLIRContext &context,
    bool profilingHarness, const std::string &profilingMeasurementsFile);

} // namespace cac
