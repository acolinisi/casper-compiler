#include "InputDesc.h"

#include "ini.h"

namespace cac {

InputDesc::InputDesc(const std::string &iniFile) {
	auto dict = parseINI(iniFile);
	for (auto &kv : dict["props"]) {
		props[kv.first] = std::stoi(kv.second);
	}
}

} // namespace cac
