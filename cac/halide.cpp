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

const std::string output_dir(".");

const CompilerLoggerFactory no_compiler_logger_factory =
  [](const std::string &, const Target &) ->
  std::unique_ptr<Internal::CompilerLogger> {
    return nullptr;
  };

std::string compileHalideKernel(const std::string &generator,
    const Target &target, const std::string &artifact,
    const std::map<std::string, std::string> &params) {
  std::string file_base_name("lib" + artifact);
  std::string function_name(artifact);
  bool build_gradient_module = false; // TODO: investigate, also 'true' breaks

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
  // We can't let compile_multitarget generate a function per target, because
  // then number of kernel functions will be num variants * num targets, and
  // for the compiler to emit a call into one of these functions, the compiler
  // would need to key by variant and by target. But this two-level keying
  // is pointless, because keying by variant should be enough -- variants
  // should encapsulate targets. Every variant defined is NOT expected to
  // be compilable for more than one target (if it is, then define the variant
  // twice with the same params, once for each target).
  std::vector<Target> variant_target{target};
  compile_multitarget(function_name, output_files, variant_target,
      module_factory, no_compiler_logger_factory);
  return output_files.at(Output::static_library);
}

std::string compileHalideRuntime(const std::vector<Target> targets) {
  assert(targets.size() > 0);
  auto target_iter = targets.begin();
  Target base_target{*target_iter};
  for (; target_iter != targets.end(); ++target_iter) {
    Target new_base_target;
    if (!base_target.get_runtime_compatible_target(*target_iter, new_base_target)) {
      throw std::runtime_error("failed to find common denominator target");
    }
    base_target = new_base_target;
  }

  std::string runtime_name("libhalide_runtime");
  std::set<Output> outputs{Output::static_library};
  std::string base_path = compute_base_path(output_dir, runtime_name, "");
  auto output_files = compute_output_files(base_target, base_path, outputs);
  compile_standalone_runtime(output_files, base_target);
  return output_files.at(Output::static_library);
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
			// The actual target is specified later during call to
			// compile_multitarget(), so this one here doesn't matter.
			const Target dummy_target{"host-no_runtime"};
			auto gen = Internal::GeneratorRegistry::create(generator,
						GeneratorContext(dummy_target));

			auto &paramInfo = gen->param_info();
			for (auto *p : paramInfo.generator_params()) {
				// TODO: need a myType() method in base class
				if (dynamic_cast<::cac::TunableGeneratorParam*>(p)) {
					halideTaskObj->impl->params.push_back(p->name);
				}
				if (dynamic_cast<::cac::InputGeneratorParam*>(p)) {
					halideTaskObj->impl->inputProps.push_back(p->name);
				}
			}
		}
	}
}

std::vector<std::string> compileHalideTasks(cac::TaskGraph &tg,
		cac::Platform &plat, cac::KnowledgeBase &kb) {
	std::vector<std::string> libs;
	std::vector<Target> targets;
	for (auto &task : tg.tasks) {
		if (task->type == cac::Task::Halide) {
			cac::HalideTask *hTask = static_cast<cac::HalideTask *>(task.get());
			const std::string &generator = task->func;
			// Compile as many variants as there are node types in the platform
			for (auto &nodeDesc : plat.nodeTypes) {
				const std::string &artifact =
					makeArtifactName(generator, nodeDesc);

				auto& params = kb.getParams(generator, nodeDesc);
				std::cerr << "params for generator " << generator << ":"
					<< std::endl;
				for (auto &kv : params) {
					std::cerr << kv.first << " = " << kv.second << std::endl;
				}
				const Target target{params.at("target")};
				targets.push_back(target);

				const std::string &lib =
				  compileHalideKernel(generator, target, artifact,
				      params);
				libs.push_back(lib);
			}
		}
	}
	const std::string &rtLib = compileHalideRuntime(targets);
	libs.push_back(rtLib);
	return libs;
}

std::vector<std::string> compileHalideTasksToProfile(cac::TaskGraph &tg,
		cac::KnowledgeBase &kb) {
	std::vector<std::string> libs;
	std::vector<Target> targets;
	for (auto &task : tg.tasks) {
		if (task->type == cac::Task::Halide) {
			cac::HalideTask *hTask = static_cast<cac::HalideTask *>(task.get());
			const std::string &generator = task->func;

			// Doing this on-demand here, but we can move this to
			// happen whenever (after generator introspection).
			std::vector<std::string> inputPropsAndParams;
			for (auto &prop: hTask->impl->inputProps)
			  inputPropsAndParams.push_back(prop);
			for (auto &param : hTask->impl->params)
			  inputPropsAndParams.push_back(param);
#if 0
			kb.drawSamples(generator, inputPropsAndParams);

			auto& samples = kb.getSamples(generator);
			unsigned i = 0;
			for (auto &sample : samples) {
#else
			unsigned validSamples = 0;
		        while (validSamples < kb.sampleCount) {
			  unsigned variantId = validSamples;
			  KnowledgeBase::ParamMap sample =
			      kb.drawSample(generator, inputPropsAndParams,
				  variantId);
#endif
				std::ostringstream artifact;
				artifact << generator << "_v" << std::to_string(variantId++);

				std::cerr << "sample params for generator " << generator
					<< "(artifact " << artifact.str() << ":" << std::endl;
				for (auto &kv : sample) {
					std::cerr << kv.first << " = " << kv.second << std::endl;
				}
				const Target target{sample.at("target")};
				targets.push_back(target);
				try {
				const std::string &lib =
				    compileHalideKernel(generator, target,
						  artifact.str(), sample);
				libs.push_back(lib);
				  ++validSamples;
				} catch(std::exception &e) {
				  std::cerr << "failed to compile variant" << std::endl;
				  continue;
				}
			}
		}
	}
	const std::string &rtLib = compileHalideRuntime(targets);
	libs.push_back(rtLib);
	return libs;
}


} // namespace cac
