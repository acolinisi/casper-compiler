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

void emitLLVMIR(cac::TaskGraph &tg, cac::Platform &plat,
    const std::string &outputFile);
int buildMLIRFromGraph(cac::TaskGraph &tg, cac::Platform &plat,
    mlir::MLIRContext &context, mlir::OwningModuleRef &module);

// TODO: split up to dedicated halide component?
void introspectHalideTasks(cac::TaskGraph &tg);
void compileHalideTasks(cac::TaskGraph &tg, cac::Platform &plat,
		cac::KnowledgeBase &kb);

} // namespace cac

#endif // CAC_BUILD_H
