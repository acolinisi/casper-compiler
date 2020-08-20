#ifndef CAC_OPTIONS_H
#define CAC_OPTIONS_H

#include <string>

namespace cac {

class Options {
public:
		Options();
		Options(int argc, char **argv);
public:
		std::string kb_filename;
		std::string kb_candidates_filename;
};

} // namespace cac

#endif // CAC_OPTIONS_H

