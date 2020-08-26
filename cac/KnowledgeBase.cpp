#include "KnowledgeBase.h"
#include "Platform.h"
#include "Options.h"

#include "knowbase.h"

#include <cassert>

namespace cac {

KnowledgeBase::KnowledgeBase() {
}

void KnowledgeBase::addNodeTypes(const std::vector<NodeDesc> &nodeTypes) {
	for (const auto &nodeType : nodeTypes) {
		this->nodeTypes.push_back(nodeType);
	}
}
void KnowledgeBase::addNodeType(NodeDesc &nodeType) {
	this->nodeTypes.push_back(nodeType);
}

std::vector<NodeDesc> KnowledgeBase::getNodeTypes() {
	return nodeTypes;
}

void KnowledgeBase::setParams(const std::string &kernelName,
		const NodeDesc &nodeDesc, ParamMap &params) {
	db[kernelName][nodeDesc.id] = params;
}

const KnowledgeBase::ParamMap&  KnowledgeBase::getParams(
		const std::string &kernelName, const NodeDesc &nodeDesc)
{
	auto platMap = db.find(kernelName);
	if (platMap != db.end()) {
		auto paramMap = platMap->second.find(nodeDesc.id);
		if (paramMap != platMap->second.end()) {
			return paramMap->second;
		}
	}
	assert(!"kernel not in KnowledgeBase");
}

} // namespace cac
