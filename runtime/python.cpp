#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


/* caller must free() */
char *concat_paths(const char *p1, const char *p2) {
	int size = strlen(p1) + 1 + strlen(p2) + 1;
	char *dest = (char *)malloc(size);
	if (!dest)
		return NULL;
	dest[0] = '\0';
	size_t remaining = size - 1;
	strcat(dest, p1);
	strcat(dest, ":");
	strcat(dest, p2);
	return dest;
}

void set_extra_python_path() {
	const char *extra_py_path = getenv("PY_KERNEL_PATH");
	if (extra_py_path) {
		wchar_t *def_py_path_w = Py_GetPath();
		char *def_py_path = Py_EncodeLocale(def_py_path_w, NULL);
		assert(def_py_path);
		printf("def_py_path: %s\n", def_py_path);

		char *py_path = concat_paths(extra_py_path, def_py_path);
		assert(py_path);
		printf("py_path: %s\n", py_path);
		PyMem_Free(def_py_path);
		wchar_t *py_path_w = Py_DecodeLocale(py_path, NULL);
		assert(py_path_w);
		free(py_path);

		Py_SetPath(py_path_w);
		PyMem_Free(py_path_w);
	}
}

extern "C" {

static wchar_t *program;

void _crt_py_init()
{
	// TODO: look into what the constructed path is
	// TODO: doesn't seem to make any difference, investigate
	program = Py_DecodeLocale("app", NULL);
	assert(program);
	Py_SetProgramName(program);

	set_extra_python_path();

	Py_Initialize();

#if 0 // attempt to append site-packages subdir to module path (no worky)
	{
		PyObject *pName, *pModule, *pValue;

		pName = PyUnicode_DecodeFSDefault("site");
		assert(pName);
		pModule = PyImport_Import(pName);
		assert(pModule);
		Py_DECREF(pName);

		pFunc = PyObject_GetAttrString(pModule, "main");
		assert(pFunc && PyCallable_Check(pFunc));
		pValue = PyObject_CallObject(pFunc, NULL);
		assert(pValue);

		Py_DECREF(pValue);
		Py_DECREF(pFunc);
		Py_DECREF(pModule);

		wchar_t *def_py_path_w = Py_GetPath();
		char *def_py_path = Py_EncodeLocale(def_py_path_w, NULL);
		assert(def_py_path);
		printf("def_py_path-site: %s\n", def_py_path);
	}
#endif
}

void _crt_py_finalize()
{
	int rc = Py_FinalizeEx();
	if (rc < 0) {
		fprintf(stderr,
			"error: failed to finalize Python interpreter: %d\n", rc);
		exit(1);
	}
	PyMem_RawFree(program);
}

// TODO: can builder access PyObject* type somhow?
void *_crt_py_alloc_obj()
{
	return Py_BuildValue("{}");
}

// TODO: can builder access PyObject* type somhow?
void _crt_py_free_obj(void *obj)
{
	PyObject *pyObj = static_cast<PyObject *>(obj);
	return Py_DECREF(pyObj);
}

// TODO: let py_func be optional
// Borrows reference to each element of args[].
int _crt_py_launch(const char *py_module, const char *py_func,
		size_t num_args, void *args[num_args])
{
	printf("py_module: %s py_func: %s\n", py_module, py_func);

	int rc;

#if 0 // TODO: mode for file path without function
	char kern_fname[1024];
	const char *py_path = getenv("PY_KERNEL_PATH");
	if (py_path) {
		kern_fname[0] = '\0';
		strcat(kern_fname, py_path);
		strcat(kern_fname, "/");
	}
	strcat(kern_fname, );
	printf("kern_fname: %s\n", kern_fname);
	assert(strlen() > 0);
	assert(strlen(py_func) > 0);

	FILE *fkern = fopen(kern_fname, "r");
	if (!fkern) {
		perror("failed to open Python kernel source code");
		exit(1);
	}
	rc = PyRun_SimpleFile(fkern, kern_fname);
	if (rc < 0) {
		fprintf(stderr, "error: failed to run Python interpreter: %d\n", rc);
		exit(1);
	}
	fclose(fkern);
#endif

	PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

	pName = PyUnicode_DecodeFSDefault(py_module);
	assert(pName);
	pModule = PyImport_Import(pName);
	Py_DECREF(pName);
	if (pModule != NULL) {
		pFunc = PyObject_GetAttrString(pModule, py_func);
		if (pFunc && PyCallable_Check(pFunc)) {
			printf("calling func\n");
			PyObject *f_args = PyTuple_New(num_args);
			for (size_t i = 0; i < num_args; ++i) {
				PyObject *arg = (PyObject *)args[i];
				Py_INCREF(arg); // don't let tuple steal the borrowed ref
				PyTuple_SetItem(f_args, i, (PyObject *)args[i]);
			}
			pValue = PyObject_CallObject(pFunc, f_args);
			Py_DECREF(f_args);
			if (pValue != NULL) {
				printf("call succeeded\n");
				Py_DECREF(pValue);
			} else {
				Py_DECREF(pFunc);
				Py_DECREF(pModule);
				PyErr_Print();
				fprintf(stderr, "error: call failed\n");
			}
		} else {
			if (PyErr_Occurred())
				PyErr_Print();
			fprintf(stderr, "error: function not found: %s\n", py_func);
		}
	} else {
		PyErr_Print();
		fprintf(stderr, "error: failed to load module '%s'\n", py_module);
	}

	return 0;
}

} // extern "C"
