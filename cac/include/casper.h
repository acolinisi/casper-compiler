#ifndef CAC_CASPER_H
#define CAC_CASPER_H

// Include complete types, so that users can use them via this one header
#include "TaskGraph.h"
#include "Platform.h"
#include "Options.h"

namespace cac {

// Emits LLVM IR to stderr
int compile(TaskGraph &tg);

// TODO: currently this func is defined by the app, but Casper lib needs to
// call it, so app needs to pass it to Casper lib as a callback pointer.
// Eventually, this will be contained within Casper lib.
class KnowledgeBase;
typedef int (TuneFunc)(TaskGraph &tg, KnowledgeBase &db);
void set_tune_func(TuneFunc &f);


} // namespace cac

#endif // CAC_CASPER_H
