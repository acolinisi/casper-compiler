#ifndef CAC_BUILD_H
#define CAC_BUILD_H

#include <string>
#include <vector>
#include <map>

namespace mlir {
  class MLIRContext;
  class OwningModuleRef;
}

namespace cac {
  class TaskGraph;
  class Platform;
}

int buildMLIRFromGraph(cac::TaskGraph &tg, cac::Platform &plat,
    mlir::MLIRContext &context, mlir::OwningModuleRef &module);

void compileHalideKernel(const std::string &generator,
    std::map<std::string, std::string> &params);
void compileHalideRuntime();

#endif // CAC_BUILD_H
