#ifndef CAC_KNOWLEDGE_BASE_H
#define CAC_KNOWLEDGE_BASE_H

#include <map>
#include <string>

namespace cac {
	class Platform;
	class NodeDesc;

	class KnowledgeBase {
	public:
		using ParamMap = std::map<std::string, std::string>;
	public:
		KnowledgeBase();
		const ParamMap& getParams(const std::string &kernelName,
				const NodeDesc &nodeDesc);
	public:
		// kernel name -> node type id -> param -> value
		std::map<std::string, std::map<unsigned, ParamMap>> params;
	};
} // namespace cac

#endif // CAC_KNOWLEDGE_BASE_H
