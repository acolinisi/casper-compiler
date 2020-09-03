#pragma once

namespace cac {

class TaskGraph;
class Platform;
class KnowledgeBase;

void introspectHalideTasks(cac::TaskGraph &tg);
void compileHalideTasks(cac::TaskGraph &tg, cac::Platform &plat,
		cac::KnowledgeBase &kb);

} // namespace cac
