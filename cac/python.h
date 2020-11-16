#pragma once

#include <cstdlib>
#include <string>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

namespace cac {
namespace py {

void init(const std::string &pythonPath);
void finalize();
void *alloc_obj();
void free_obj(void *obj);
int launch(const std::string &py_module, const std::string &py_func,
		size_t num_args, PyObject *args[num_args], PyObject **ret);

} // namespace py
} // namespace cac
