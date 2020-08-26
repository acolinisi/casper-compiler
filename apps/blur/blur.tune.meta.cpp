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
void buildKB(graph_t &KB,
		const char *modelFile, const char *modelCPFile) {
	// add hardware
	vertex_descriptor_t hardware0 = boost::add_vertex(KB);
	CPU_t *i7_cpu= new CPU_t(4, 50000000000);
	i7_cpu->type = "CPU_t";
	i7_cpu->id = 0;
	i7_cpu->node_type = 0;
	KB[hardware0].is_hardware = true;
	KB[hardware0].hardware = i7_cpu;
	KB[hardware0].id = i7_cpu->id;

	// add step
	vertex_descriptor_t step0 = boost::add_vertex(KB);
	Blur_t *Blur = new Blur_t();
	Blur->type = "Blur_t";
	Blur->id = 1;
	Blur->name = "halide_blur";
	KB[step0].is_step = true;
	KB[step0].step = Blur;
	KB[step0].id = Blur->id;

	// add performance model
	std::pair<edge_descriptror_t, bool> h0s0 = boost::add_edge(hardware0, step0, KB);
	MLP_t *halide_i7 = new MLP_t((char *)modelFile, (char *)modelCPFile);
	halide_i7->type = "MLP_t";
	halide_i7->id = 2;
	halide_i7->src_id = KB[hardware0].id;
	halide_i7->dst_id = KB[step0].id;
	KB[h0s0.first].is_performance_model = true;
	KB[h0s0.first].performance_model = halide_i7;

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
	buildKB(kbGraph, modelFile, modelCPFile);

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
		for (auto &plat: platforms) {
			vertex_descriptor_t &platV = plat.first;
			NodeDesc &nodeDesc = plat.second;

			std::vector<float> variant =
				select_variant(kbGraph, taskV, platV, candidatesFile, 1024);

			// TODO: generalize to any kernel: introspect the generator?
			KnowledgeBase::ParamMap params{
				{ "p1", std::to_string((int)variant[0]) },
				{ "p2", std::to_string((int)variant[1]) },
				{ "p3", std::to_string((int)variant[2]) },
				{ "p4", std::to_string((int)variant[3]) },
			};
			db.setParams(taskObj.func, nodeDesc, params);
		}
	}

	return 0;
}

} // namespace cac
