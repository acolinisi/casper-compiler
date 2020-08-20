#ifndef CAC_KNOWLEDGE_BASE_H
#define CAC_KNOWLEDGE_BASE_H

#include <map>
#include <string>

#include "knowbase.h"

namespace cac {
	class Platform;
	class NodeDesc;
	class Options;

	class KnowledgeBase {
	public:
		using ParamMap = std::map<std::string, std::string>;
	public:
		KnowledgeBase(Options &opts);
		const ParamMap& getParams(const std::string &kernelName,
				const NodeDesc &nodeDesc);
	public:
		// kernel name -> node type id -> param -> value
		std::map<std::string, std::map<unsigned, ParamMap>> params;
		graph_t kb_graph;
	};
} // namespace cac

#endif // CAC_KNOWLEDGE_BASE_H
