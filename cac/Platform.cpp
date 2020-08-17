#include "Platform.h"

namespace cac {

NodeDesc::NodeDesc(unsigned id) : id(id) {}

Platform::Platform() : nodeTypes{0} {}
Platform::Platform(std::vector<NodeDesc> nodeTypes) : nodeTypes(nodeTypes) {}

} // namespace cac
