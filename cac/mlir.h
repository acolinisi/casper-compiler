#pragma once

#include <string>

namespace cac {

class TaskGraph;
class Platform;

void emitLLVMIR(cac::TaskGraph &tg, cac::Platform &plat,
    bool profilingHarness, const std::string &outputFile);

} // namespace cac
