#pragma once

#include <string>

namespace cac {

class TaskGraph;
class Platform;

void emitLLVMIR(const std::string &outputFile, cac::TaskGraph &tg,
        cac::Platform &plat,
        bool profilingHarness, const std::string &profilingMeasurementsFile);

} // namespace cac
