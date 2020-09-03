#include "CasperHalide.h"
#include "TaskGraph.h"
#include "TaskGraphImpl.h"
#include "Platform.h"
#include "KnowledgeBase.h"

#include <Halide.h>
#include <Halide/Generator.h>

#include <set>
#include <vector>
#include <map>

using namespace Halide;

// Don't bring in Halide::Internal:: namespace in order to be explicit about
// types not exposed in Halide:: namespace, could ask to expose them.
#if 0
using namespace Halide::Internal;
#endif

namespace {

std::vector<std::string> split_string(const std::string &source, const std::string &delim) {
    std::vector<std::string> elements;
    size_t start = 0;
    size_t found = 0;
    while ((found = source.find(delim, start)) != std::string::npos) {
        elements.push_back(source.substr(start, found - start));
        start = found + delim.size();
    }

    // If start is exactly source.size(), the last thing in source is a
    // delimiter, in which case we want to add an empty string to elements.
    if (start <= source.size()) {
        elements.push_back(source.substr(start, std::string::npos));
    }
    return elements;
}

std::string extract_namespaces(const std::string &name, std::vector<std::string> &namespaces) {
    namespaces = split_string(name, "::");
    std::string result = namespaces.back();
    namespaces.pop_back();
    return result;
}

std::string compute_base_path(const std::string &output_dir,
                              const std::string &function_name,
                              const std::string &file_base_name) {
    std::vector<std::string> namespaces;
    std::string simple_name = extract_namespaces(function_name, namespaces);
    std::string base_path = output_dir + "/" + (file_base_name.empty() ? simple_name : file_base_name);
    return base_path;
}

std::map<Output, std::string> compute_output_files(const Target &target,
                                                   const std::string &base_path,
                                                   const std::set<Output> &outputs) {
    std::map<Output, Internal::OutputInfo> output_info = Internal::get_output_info(target);

    std::map<Output, std::string> output_files;
    for (auto o : outputs) {
        output_files[o] = base_path + output_info.at(o).extension;
    }
    return output_files;
}

std::string makeArtifactName(const std::string &generator,
    const cac::NodeDesc &nodeDesc)
{
      return generator + "_v" + std::to_string(nodeDesc.id);
}

const Target target{"host-no_runtime"};
const std::string output_dir(".");

const CompilerLoggerFactory no_compiler_logger_factory =
  [](const std::string &, const Target &) ->
  std::unique_ptr<Internal::CompilerLogger> {
    return nullptr;
  };

void compileHalideKernel(const std::string &generator,
    const std::string &artifact,
    const std::map<std::string, std::string> &params) {
  std::string file_base_name("lib" + artifact);
  std::string function_name(artifact);
  bool build_gradient_module = false; // TODO: investigate, also 'true' breaks
  std::vector<Target> targets{target};

  std::string base_path = compute_base_path(output_dir, function_name,
      file_base_name);
  Internal::GeneratorParamsMap generator_args;
  for (auto &param : params) {
    generator_args[param.first] = param.second;
  }
  std::set<Output> outputs{
    Output::static_library,
    Output::stmt, // for debugging
  };
  auto output_files = compute_output_files(target, base_path, outputs);
  auto module_factory = [&generator, &generator_args, build_gradient_module]
    (const std::string &name, const Target &target) -> Module {
	  // NOTE: generator objects are single-use, so recreate for each build
      auto gen = Internal::GeneratorRegistry::create(generator,
	  				GeneratorContext(target));
      gen->set_generator_param_values(generator_args);
      return build_gradient_module ?
		  gen->build_gradient_module(name) : gen->build_module(name);
  };
  compile_multitarget(function_name, output_files, targets, module_factory,
      no_compiler_logger_factory);
}

void compileHalideRuntime() {
  std::string runtime_name("libhalide_runtime");
  std::set<Output> outputs{Output::static_library};
  std::string base_path = compute_base_path(output_dir, runtime_name, "");
  auto output_files = compute_output_files(target, base_path, outputs);
  compile_standalone_runtime(output_files, target);
}

} // namespace anon

namespace cac {

// Populates fields in Halide with tunable parameter names.
void introspectHalideTasks(cac::TaskGraph &tg) {
	for (auto &task : tg.tasks) {
		if (task->type == cac::Task::Halide) {
			cac::HalideTask *halideTaskObj =
				static_cast<cac::HalideTask *>(task.get());
			const std::string &generator = task->func;

			// Note: we can't save the generator in task obj and re-use
			// it for compilation, because generators are single-use,
			// whereas we need to compile per platform.
			auto gen = Internal::GeneratorRegistry::create(generator,
						GeneratorContext(target));

			auto &paramInfo = gen->param_info();
			for (auto *p : paramInfo.generator_params()) {
				// TODO: need a myType() method in base class
				if (dynamic_cast<::cac::TunableGeneratorParam*>(p)) {
					halideTaskObj->impl->params.push_back(p->name);
				}
			}
		}
	}
}

void compileHalideTasks(cac::TaskGraph &tg, cac::Platform &plat,
		cac::KnowledgeBase &kb) {
	for (auto &task : tg.tasks) {
		if (task->type == cac::Task::Halide) {
			cac::HalideTask *hTask = static_cast<cac::HalideTask *>(task.get());
			// Compile as many variants as there are node types in the platform
			for (auto &nodeDesc : plat.nodeTypes) {
				const std::string &generator = task->func;
				const std::string &artifact =
					makeArtifactName(generator, nodeDesc);

				auto& params = kb.getParams(generator, nodeDesc);
				std::cerr << "params for generator " << generator << ":"
					<< std::endl;
				for (auto &kv : params) {
					std::cerr << kv.first << " = " << kv.second << std::endl;
				}

				compileHalideKernel(generator, artifact, params);
			}
		}
	}
	compileHalideRuntime();
}


} // namespace cac
