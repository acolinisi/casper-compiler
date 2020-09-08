#pragma once

#include <map>
#include <string>

namespace cac {

class InputDesc {
public:
	InputDesc(const std::string &iniFile);
public:
	// property name -> value
	// TODO: generalize to support more than int properties
	std::map<std::string, int> props;
};

} // namespace cac
