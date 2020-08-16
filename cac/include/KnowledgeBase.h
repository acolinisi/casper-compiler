#ifndef CAC_KNOWLEDGE_BASE_H
#define CAC_KNOWLEDGE_BASE_H

#include <map>
#include <string>

namespace cac {
	class KnowledgeBase {
	public:
		using ParamMap = std::map<std::string, std::string>;
	public:
		KnowledgeBase();
		const ParamMap& getParams(const std::string &kernelName);
	public:
		std::map<std::string, ParamMap> params;
	};
} // namespace cac

#endif // CAC_KNOWLEDGE_BASE_H
