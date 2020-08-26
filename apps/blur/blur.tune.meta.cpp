// Mockup for what the autotuner flow will do: profile, train,
// and evaluat the performance predictor model.

#include "casper.h"
#include "KnowledgeBase.h"

#include "knowbase.h"

//#include <fstream>
#include <string>

using namespace cac;

namespace cac {

int tune(TaskGraph &tg, KnowledgeBase &db)
{
	// TODO: cache DB object, and make all steps here incremental

	// Build artifacts
	const char *modelFile = "tf_models/model.pb";
	const char *modelCPFile = "tf_models/my-model.ckpt";
	const char *candidatesFile = "halide_blur_i7_candidates.small.csv";

	graph_t &kbGraph = db.kbGraph;

	const std::vector<vertex_descriptor_t> &platforms =
		db.getNodeTypeVertices();

	// Create vertices for tasks (aka. steps) in the knowledge base
	typedef std::pair<vertex_descriptor_t, Task&> TaskPair;
	std::vector<TaskPair> tasks;
	int id = 0;
	for (const auto &task : tg.tasks) {
		// TODO: For now, we tune all Halide kernels for all platforms
		if (task->type != Task::Halide) {
			continue;
		}

		vertex_descriptor_t step = boost::add_vertex(kbGraph);
		Step_t *s = new Step_t();
		s->type = "Step_t";
		s->id = id++;
		s->name = task->func;
		kbGraph[step].is_step = true;
		kbGraph[step].step = s;
		kbGraph[step].id = s->id;

		tasks.push_back(TaskPair{step, *task});
	}

	// Create edges for perf models in the knowledge base
	for (auto &taskPair : tasks) {
		vertex_descriptor_t step = taskPair.first;
		for (auto &platform : platforms) {
			const std::pair<edge_descriptror_t, bool> edge =
				boost::add_edge(platform, step, kbGraph);

			// TODO: re-using same mocked up model files for now
			MLP_t *m = new MLP_t((char *)modelFile, (char *)modelCPFile);
			m->type = "MLP_t";
			m->id = 2;
			m->src_id = kbGraph[platform].id;
			m->dst_id = kbGraph[step].id;
			kbGraph[edge.first].is_performance_model = true;
			kbGraph[edge.first].performance_model = m;
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
		for (auto &platV: platforms) {
			NodeDesc nodeDesc{kbGraph[platV].hardware->node_type};

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
