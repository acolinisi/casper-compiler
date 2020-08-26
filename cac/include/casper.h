#ifndef CAC_CASPER_H
#define CAC_CASPER_H

// Include complete types, so that users can use them via this one header
#include "TaskGraph.h"
#include "Options.h"

namespace cac {

int compile(TaskGraph &tg);
void compileThrow(TaskGraph &tg);

} // namespace cac

#endif // CAC_CASPER_H
