#include "KnowledgeBase.h"
#include "Platform.h"

#include <cassert>

namespace cac {

KnowledgeBase::KnowledgeBase() {
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
