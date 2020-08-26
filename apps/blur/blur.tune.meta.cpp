// Mockup for what the autotuner flow will do: profile, train,
// and evaluat the performance predictor model.

#include "casper.h"
#include "KnowledgeBase.h"

#include "knowbase.h"

//#include <fstream>
#include <string>

using namespace cac;

namespace {

// TODO: this should be built from the TaskGraph object
void buildKB(graph_t &KB, TaskGraph &tg,
		const char *modelFile, const char *modelCPFile) {
	// add hardware
	std::vector<vertex_descriptor_t> platforms;

	vertex_descriptor_t hardware0 = boost::add_vertex(KB);
	CPU_t *i7_cpu= new CPU_t(4, 50000000000);
	i7_cpu->type = "CPU_t";
	i7_cpu->id = 0;
	i7_cpu->node_type = 0;
	KB[hardware0].is_hardware = true;
	KB[hardware0].hardware = i7_cpu;
	KB[hardware0].id = i7_cpu->id;
	platforms.push_back(hardware0);

	vertex_descriptor_t hardware1 = boost::add_vertex(KB);
	CPU_t *dummy_cpu= new CPU_t(8, 20000000000);
	dummy_cpu->type = "CPU_t";
	dummy_cpu->id = 1;
	dummy_cpu->node_type = 1;
	KB[hardware1].is_hardware = true;
	KB[hardware1].hardware = dummy_cpu;
	KB[hardware1].id = dummy_cpu->id;
	platforms.push_back(hardware1);

	std::map<std::string, vertex_descriptor_t> steps;
	int id = 0;
	for (auto &task : tg.tasks) {
		// TODO: For now, we tune all Halide kernels for all platforms
		if (task->type != Task::Halide) {
			continue;
		}

		vertex_descriptor_t step = boost::add_vertex(KB);
		Step_t *s = new Step_t();
		s->type = "Step_t";
		s->id = id++;
		s->name = task->func;
		KB[step].is_step = true;
		KB[step].step = s;
		KB[step].id = s->id;
		steps[s->name] = step;

		for (auto &platform : platforms) {
			const std::pair<edge_descriptror_t, bool> edge =
				boost::add_edge(platform, step, KB);

			// TODO: re-using same mocked up model files for now
			MLP_t *m = new MLP_t((char *)modelFile, (char *)modelCPFile);
			m->type = "MLP_t";
			m->id = 2;
			m->src_id = KB[hardware0].id;
			m->dst_id = KB[step].id;
			KB[edge.first].is_performance_model = true;
			KB[edge.first].performance_model = m;
		}
	}

	// TODO: free memory
}

} // namespace anon

namespace cac {

int tune(TaskGraph &tg, KnowledgeBase &db)
{
	// TODO: cache DB object, and make all steps here incremental

	// Build artifacts
	const char *modelFile = "tf_models/model.pb";
	const char *modelCPFile = "tf_models/my-model.ckpt";
	const char *candidatesFile = "halide_blur_i7_candidates.small.csv";

	graph_t &kbGraph = db.kbGraph;
	buildKB(kbGraph, tg, modelFile, modelCPFile);

	// Enumerate platforms (and cross-check tasks) defined in KB graph
	typedef std::pair<vertex_descriptor_t, NodeDesc> PlatPair;
	typedef std::pair<vertex_descriptor_t, Task&> TaskPair;
	std::vector<PlatPair> platforms;
	std::vector<TaskPair> tasks;
	auto v_it_range = boost::vertices(kbGraph);
	for (auto it = v_it_range.first; it != v_it_range.second; ++it) {
		if (kbGraph[*it].is_hardware) {
			NodeDesc nodeDesc{kbGraph[*it].hardware->node_type};
			db.addNodeType(nodeDesc);
			platforms.push_back(PlatPair{*it, nodeDesc});
		}
		// TODO: this logic will change if we do build KB per app
		if (kbGraph[*it].is_step) { // TODO: is_kernel?
			for (auto &task: tg.tasks) {
				assert(kbGraph[*it].step);
				if (task->func == kbGraph[*it].step->name) {
					tasks.push_back(TaskPair{*it, *task});
				}
			}
		}
	}

	std::cout << "Tuning variant parameters for: " << tasks.size() << " tasks, "
		<< platforms.size() << " platforms..." << std::endl;

	for (auto &task: tasks) {
		vertex_descriptor_t &taskV = task.first;
		Task &taskObj = task.second;
		if (taskObj.type != Task::Halide) {
			continue; // TODO: support variants for other task types too
		}
		HalideTask *halideTaskObj = static_cast<HalideTask*>(&taskObj);
		for (auto &plat: platforms) {
			vertex_descriptor_t &platV = plat.first;
			NodeDesc &nodeDesc = plat.second;

			std::map<std::string, float> variant =
				select_variant(kbGraph, taskV, platV, candidatesFile, 1024);
			KnowledgeBase::ParamMap params;
			for (const auto &param : halideTaskObj->params) {
				params[param] = std::to_string((int)variant[param]);
			}

			db.setParams(halideTaskObj->func, nodeDesc, params);
		}
	}

	return 0;
}

} // namespace cac
