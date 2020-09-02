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
  class KnowledgeBase;

int buildMLIRFromGraph(cac::TaskGraph &tg, cac::Platform &plat,
    cac::KnowledgeBase &kb,
    mlir::MLIRContext &context, mlir::OwningModuleRef &module);

// TODO: split up to dedicated halide component?

void compileHalideKernel(const std::string &generator,
    const std::string &artifact,
    const std::map<std::string, std::string> &params);
void compileHalideRuntime();

} // namespace cac

#endif // CAC_BUILD_H
