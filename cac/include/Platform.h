#ifndef CAC_PLATFORM_H
#define CAC_PLATFORM_H

#include <vector>

namespace cac {
	class NodeDesc {
	public:
		NodeDesc(unsigned id);
	public:
		unsigned id;
	};

	class Platform {
	public:
		Platform();
		Platform(std::vector<NodeDesc> nodeTypes);
	public:
		std::vector<NodeDesc> nodeTypes;
	};
} // namespace cac

#endif // CAC_PLATFORM_H
