#ifndef CAC_CASPER_H
#define CAC_CASPER_H

// Include complete types, so that users can use them via this one header
#include "TaskGraph.h"
#include "Platform.h"
#include "Options.h"

namespace cac {

// Emits LLVM IR to stderr
int compile(TaskGraph &tg);
int compile(TaskGraph &tg, Platform &plat);
int compile(TaskGraph &tg, Platform &plat, Options &opts);

} // namespace cac

#endif // CAC_CASPER_H
