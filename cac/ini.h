#pragma once

#include <map>
#include <string>

namespace cac {

// section -> key -> value
typedef std::map<std::string, std::map<std::string, std::string>> INIDict;

INIDict parseINI(const std::string &iniFile);

} // namespace cac
