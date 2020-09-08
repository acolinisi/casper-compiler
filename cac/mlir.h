#pragma once

#include <string>
#include <vector>

namespace cac {

class TaskGraph;
class Platform;

void emitLLVMIR(const std::string &outputFile, cac::TaskGraph &tg,
        std::vector<unsigned> variantIds,
        bool profilingHarness, const std::string &profilingMeasurementsFile);

} // namespace cac
