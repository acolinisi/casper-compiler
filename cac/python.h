#pragma once

#include <cstdlib>
#include <string>

namespace cac {
namespace py {

void init(const std::string &pythonPath);
void finalize();
void *alloc_obj();
void free_obj(void *obj);
int launch(const std::string &py_module, const std::string &py_func,
		size_t num_args, void *args[num_args]);

} // namespace py
} // namespace cac
