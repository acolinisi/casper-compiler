#include "KnowledgeBase.h"
#include "Platform.h"
#include "Options.h"
#include "ini.h"

#include "knowbase.h"

#include <cassert>
#include <iostream>
#include <fstream>
#include <string.h>

namespace cac {

KnowledgeBase::KnowledgeBase(const std::string &samplesFilename) :
	// TODO: KnowledgeBase is responsible for figuring out sample count
	sampleCount(16),
	samplesFilename(samplesFilename)
{
	std::ofstream samplesFile(samplesFilename);
	samplesFile << "task,variant,param,value" << std::endl;
}

void KnowledgeBase::loadPlatform(const std::string &iniFile) {
	const auto &dict = parseINI(iniFile);
	unsigned nodeTypeId = 0;
	for (const auto &sectPair : dict) {
		const auto &sect = sectPair.second;

		std::string type;
		{
		const auto &it = sect.find("type");
		assert(it != sect.end());
		type = it->second;
		}

		vertex_descriptor_t v = boost::add_vertex(kbGraph);
		Hardware_t *hw;
		int id = nodeTypeId++;

		if (type == "cpu") {
			unsigned cores, freq_hz;
			{
			const auto &it = sect.find("cores");
			assert(it != sect.end());
			cores = std::stoul(it->second);
			}

			{
			const auto &it = sect.find("freq_hz");
			assert(it != sect.end());
			freq_hz = std::stoul(it->second);
			}

			CPU_t *cpu= new CPU_t(cores, freq_hz);
			// TODO: cleanup memory
			cpu->type = "CPU_t";
			cpu->id = id;
			cpu->node_type = id;
			hw = cpu;
		} else {
			std::ostringstream msg;
			msg << "unsupported node type: " << type;
			throw std::runtime_error{msg.str()};
		}

		kbGraph[v].is_hardware = true;
		kbGraph[v].hardware = hw;
		kbGraph[v].id = id;
	}
}

std::vector<NodeDesc> KnowledgeBase::getNodeTypes() {
	std::vector<NodeDesc> nodeTypes;
	auto v_it_range = boost::vertices(kbGraph);
	for (auto it = v_it_range.first; it != v_it_range.second; ++it) {
		if (kbGraph[*it].is_hardware) {
			NodeDesc nodeDesc{kbGraph[*it].hardware->node_type};
			nodeTypes.push_back(nodeDesc);
		}
	}
	return nodeTypes;
}
std::vector<vertex_descriptor_t> KnowledgeBase::getNodeTypeVertices() {
	std::vector<vertex_descriptor_t> vertices;
	auto v_it_range = boost::vertices(kbGraph);
	for (auto it = v_it_range.first; it != v_it_range.second; ++it) {
		if (kbGraph[*it].is_hardware)
			vertices.push_back(*it);
	}
	return vertices;
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

void KnowledgeBase::drawSamples(const std::string &generator,
		std::vector<std::string> paramNames)
{
	std::ofstream samplesFile(samplesFilename, std::ios_base::app);
	for (int i = 0; i < sampleCount; ++i) {
		ParamMap params;
		for (auto &paramName : paramNames) {
#if 0
			params[paramName] = std::to_string(1 + rand() % 8); // TODO: <random>
#else
			params[paramName] = std::to_string(i + 1);
#endif
			std::cout << "gen " << generator << " param " << paramName
				<< std::endl;
			samplesFile << generator << "," << i << ","
				<< paramName << "," << params[paramName] << std::endl;
		}
		samples[generator].insert(params);
		std::cout << "gen " << generator << " size "
			<< samples[generator].size() << std::endl;
	}
}

std::set<KnowledgeBase::ParamMap>& KnowledgeBase::getSamples(
		const std::string &generator) {
	auto it = samples.find(generator);
	if (it == samples.end()) {
		std::ostringstream msg;
		msg << "kernel '" << generator << "' not in knowledge base";
		throw std::runtime_error(msg.str());
	}
	std::cout << "get gen " << generator << " size "
		<< it->second.size() << std::endl;
	return it->second;
}

} // namespace cac
