#ifndef CAC_KNOWLEDGE_BASE_H
#define CAC_KNOWLEDGE_BASE_H

#include <map>
#include <set>
#include <string>
#include <vector>

#include "knowbase.h"

namespace cac {
	class Platform;
	class NodeDesc;

	class KnowledgeBase {
	public:
		using ParamMap = std::map<std::string, std::string>;
		using DB = std::map<std::string, std::map<unsigned, ParamMap>>;
	public:
		KnowledgeBase(const std::string &samplesFilename);
		void loadPlatform(const std::string &iniFile);
		std::vector<NodeDesc> getNodeTypes();
		std::vector<vertex_descriptor_t> getNodeTypeVertices();
		void setParams(const std::string &kernelName,
				const NodeDesc &nodeDesc, ParamMap &params);
		const ParamMap& getParams(const std::string &kernelName,
				const NodeDesc &nodeDesc);
		void drawSamples(const std::string &generator,
				std::vector<std::string> paramNames);
		std::set<ParamMap> &getSamples(const std::string &generator);
	public:
		// kernel name -> node type id -> param -> value
		DB db;
		graph_t kbGraph;
		// kernel name -> { param -> value, ... }
		std::map<std::string, std::set<ParamMap>> samples;
		unsigned sampleCount; // variants of each kernel to profile
		std::string samplesFilename;
	};
} // namespace cac

#endif // CAC_KNOWLEDGE_BASE_H
