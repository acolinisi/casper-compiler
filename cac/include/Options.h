#ifndef CAC_OPTIONS_H
#define CAC_OPTIONS_H

#include <string>
#include <memory>

namespace cac {

class Options {
public:
		Options();
		~Options();
		void parse(int argc, char **argv);
		void parseOrExit(int argc, char **argv);
public:
		std::string llOutputFile;
		std::string buildArgsFile;
		std::string platformFile;
		std::string modelsDir;
		std::string inputDescFile;

		bool profilingHarness;
		std::string profilingMeasurementsFile;
		std::string profilingSamplesFile;

		// TODO: These will change eventually: will be generated during the
		// compilation flow, and models are per task, not per app.
		std::string modelFile;
		std::string modelCPFile;
		std::string candidatesFile;

		bool profile;
private:
		class Impl;
		std::unique_ptr<Impl> impl;
};

} // namespace cac

#endif // CAC_OPTIONS_H

