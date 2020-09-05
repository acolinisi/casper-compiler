#pragma once

namespace cac {

class TaskGraph;
class Platform;
class KnowledgeBase;

void introspectHalideTasks(cac::TaskGraph &tg);
std::vector<std::string> compileHalideTasks(cac::TaskGraph &tg,
		cac::Platform &plat, cac::KnowledgeBase &kb);
std::vector<std::string> compileHalideTasksToProfile(cac::TaskGraph &tg,
		cac::KnowledgeBase &kb);

} // namespace cac
