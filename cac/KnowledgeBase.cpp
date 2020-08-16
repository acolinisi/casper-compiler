#include "KnowledgeBase.h"
#include <cassert>

namespace cac {

KnowledgeBase::KnowledgeBase() {
	// Dummy mockup
	KnowledgeBase::ParamMap blur_params = {
		{ "p1", "1" },
		{ "p2", "2" },
		{ "p3", "3" },
		{ "p4", "4" },
	};

	params["blur"] = blur_params;
}

const KnowledgeBase::ParamMap&  KnowledgeBase::getParams(
		const std::string &kernelName)
{
	auto item = params.find(kernelName);
	if (item != params.end()) {
		return item->second;
	} else {
		return params[kernelName]; // inserts
	}
}

} // namespace cac
