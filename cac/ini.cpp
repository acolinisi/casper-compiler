#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <string.h>

#include <iostream>

#include "ini.h"

namespace {
	void trim(std::string &s) {
		auto pred = [](int ch) { return !std::isspace(ch); };
		s.erase(s.begin(), std::find_if(s.begin(), s.end(), pred));
		s.erase(std::find_if(s.rbegin(), s.rend(), pred).base(), s.end());
	}
} // namespace anon

namespace cac {

INIDict parseINI(const std::string &iniFile) {
	INIDict dict;

	std::ifstream fin(iniFile);
	if (!fin.is_open()) {
		std::ostringstream msg;
		msg << "failed to open file '" << iniFile << "': " << strerror(errno);
		throw std::runtime_error{msg.str()};
	}

	std::string line, section;
	size_t p, lineNum = 0;
	while (std::getline(fin, line)) {
		trim(line);
		if (line.size() == 0 || *line.begin() == '#') // skip comments
			continue;
		std::cerr << line << std::endl;
		if (line.size() > 2 &&
				*line.begin() == '[' && *(--line.end()) == ']') {
			std::string sect{++line.begin(), --line.end()};
			trim(sect);
			dict.insert(INIDict::value_type{sect, {}});
			section = sect;
		}
		else if (line.size() > 2 && (p = line.find('=')) != std::string::npos) {
			std::string key{line, 0, p}, value{line, p + 1};
			trim(key); trim(value);
			dict[section][key] = value;
		} else {
			std::ostringstream msg;
			msg << "failed to parse line '" << iniFile << ":" << lineNum << ": "
				<< strerror(errno);
			throw std::runtime_error{msg.str()};
		}
		++lineNum;
	}
	return dict;
}

} // namespace cac
