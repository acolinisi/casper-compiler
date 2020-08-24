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
void build_kb_graph(graph_t &KB,
		const char *model_fname, const char *model_cp_fname) {
	// add hardware
	vertex_descriptor_t hardware0 = boost::add_vertex(KB);
	CPU_t *i7_cpu= new CPU_t(4, 50000000000);
	i7_cpu->type = "CPU_t";
	i7_cpu->id = 0;
	KB[hardware0].is_hardware = true;
	KB[hardware0].hardware = i7_cpu;
	KB[hardware0].id = i7_cpu->id;

	// add step
	vertex_descriptor_t step0 = boost::add_vertex(KB);
	Blur_t *Blur = new Blur_t();
	Blur->type = "Blur_t";
	Blur->id = 1;
	KB[step0].is_step = true;
	KB[step0].step = Blur;
	KB[step0].id = Blur->id;

	// add performance model
	std::pair<edge_descriptror_t, bool> h0s0 = boost::add_edge(hardware0, step0, KB);
	MLP_t *halide_i7 = new MLP_t((char *)model_fname, (char *)model_cp_fname);
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
	char *model_fname = "tf_models/model.pb";
	char *model_cp_fname = "tf_models/my-model.ckpt";
	const char *candidates_fname = "halide_blur_i7_candidates.small.csv";
#if 0
	const char *variants_fname = "blur.variants";
#endif

	const char *kb_fname = "blur"; // temporary file

	graph_t kb_graph;
	build_kb_graph(kb_graph, model_fname, model_cp_fname);

	// query
	// float dimension = 32768;
	// std::vector<float> schedule = {1024, 4, 2, 2};
	// std::cout << halide_i7.eval(dimension, schedule).exec_time << std::endl;

	// TODO: shouldn't need to save and load to get vertices (use vtx iterator)
	save_graph(kb_graph, kb_fname);
	std::vector<vertex_descriptor_t> vertices = load_graph(kb_graph, kb_fname);

	// Finds the best variant
	std::vector<float> variant = select_variant(1024, kb_graph, vertices,
	        candidates_fname);

	const char *kern_name = "halide_blur";
	const NodeDesc nodeDesc{1}; // TODO: get from HW info in kb_graph
	// TODO: generalize to any kernel: introspect the generator?
	KnowledgeBase::ParamMap params{
		{ "p1", std::to_string((int)variant[0]) },
		{ "p2", std::to_string((int)variant[1]) },
		{ "p3", std::to_string((int)variant[2]) },
		{ "p4", std::to_string((int)variant[3]) },
   	};
	db.setParams(kern_name, nodeDesc, params);

#if 0
	std::ofstream variants_fout(opts.variants_fname);
	variants_fout << "halide_blur"; // kernel ID (from metaprogram)
	for (float param : variant)
	    variants_fout << param;
	variants_fout << std::endl;
	variants_fout.close();
#endif

	// query
	// struct metadata_t metadata;
	// metadata.dimension = 32768;
	// metadata.schedule = {1024, 4, 2, 2};
	// metadata.schedule = {16, 64, 32, 16};

	// kb_graph[boost::edge(vertices[0], vertices[1], kb_graph).first].performance_model->eval(metadata).exec_time

	return 0;
}

} // namespace cac
