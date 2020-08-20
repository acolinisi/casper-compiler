#include "KnowledgeBase.h"
#include "Platform.h"
#include "Options.h"

#include "knowbase.h"

#include <cassert>

namespace cac {

KnowledgeBase::KnowledgeBase(Options &opts) {

#if 0 // TODO: not here
    std::vector<vertex_descriptor_t> vertices =
	load_graph(kb_graph, opts.kb_filename);

    // Finds the best variant
    std::vector<float> variant = select_variant(1024, kb_graph, vertices,
	    opts.kb_candidates_filename);

    // query
    // struct metadata_t metadata;
    // metadata.dimension = 32768;
    // metadata.schedule = {1024, 4, 2, 2};
    // metadata.schedule = {16, 64, 32, 16};

    // kb_graph[boost::edge(vertices[0], vertices[1], kb_graph).first].performance_model->eval(metadata).exec_time

#endif

	// Dummy mockup. Map: kernel -> node type id -> param -> value
	std::map<std::string, std::map<unsigned, ParamMap>> dummyParams =
	{
		{ "halide_bright", { { 0, { }, }, } },
		{ "halide_blur",
			{
				{ 0, // very slow (>1m)
					{
						{ "p1", "1" },
						{ "p2", "1" },
						{ "p3", "1" },
						{ "p4", "1" },
					},
				},
				{ 1, // very fast (<14s)
					{
						{ "p1", "2" },
						{ "p2", "8" },
						{ "p3", "8" },
						{ "p4", "1" },
					},
				},
			},
		},
	};
	params = dummyParams;
}

const KnowledgeBase::ParamMap&  KnowledgeBase::getParams(
		const std::string &kernelName, const NodeDesc &nodeDesc)
{
	auto platMap = params.find(kernelName);
	if (platMap != params.end()) {
		auto paramMap = platMap->second.find(nodeDesc.id);
		if (paramMap != platMap->second.end()) {
			return paramMap->second;
		}
	}
	assert(!"kernel not in KnowledgeBase");
}

} // namespace cac
