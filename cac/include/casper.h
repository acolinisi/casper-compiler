#ifndef CAC_CASPER_H
#define CAC_CASPER_H

#include "TaskGraph.h"
#include "Platform.h"

namespace cac {

// Emits LLVM IR to stderr
int compile(TaskGraph &tg);
int compile(TaskGraph &tg, Platform &plat);

} // namespace cac

#endif // CAC_CASPER_H
