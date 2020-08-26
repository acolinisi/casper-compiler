#ifndef CAC_CASPER_H
#define CAC_CASPER_H

// Include complete types, so that users can use them via this one header
#include "TaskGraph.h"
#include "Options.h"

namespace cac {

// throws on failure
void compile(TaskGraph &tg);

// compile() wrapped in a try/catch that returns non-zero on failure
int tryCompile(TaskGraph &tg);

} // namespace cac

#endif // CAC_CASPER_H
