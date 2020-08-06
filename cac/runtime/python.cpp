#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#if 0
/* caller must free() */
char *concat_paths(const char *p1, const char *p2) {
	int size = strlen(p1) + 1 + strlen(p2) + 1;
	char *dest = malloc(size);
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
	const char *extra_py_path = getenv("EXTRA_PYTHON_PATH");
	if (extra_py_path) {
		wchar_t *def_py_path_w = Py_GetPath();
		char *def_py_path = Py_EncodeLocale(def_py_path_w, NULL);
		assert(def_py_path);

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
#endif

extern "C" {

int launch_python(const char *py_file)
{
	printf("python\n");

	// TODO: look into what the constructed path is
	wchar_t *program = Py_DecodeLocale("blur", NULL);
	assert(program);
	Py_SetProgramName(program);
	//set_extra_python_path();

	Py_Initialize();

	int rc;

	char kern_fname[1024];
	const char *py_path = getenv("PY_KERNEL_PATH");
	if (py_path) {
		kern_fname[0] = '\0';
		strcat(kern_fname, py_path);
		strcat(kern_fname, "/");
	}
	strcat(kern_fname, py_file);
	printf("kern_fname: %s\n", kern_fname);

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

	rc = Py_FinalizeEx();
	if (rc < 0) {
		fprintf(stderr,
			"error: failed to finalize Python interpreter: %d\n", rc);
		exit(1);
	}

	PyMem_RawFree(program);
	return 0;
}

} // extern "C"
