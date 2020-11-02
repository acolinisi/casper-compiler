#include <stdint.h>
#include <stdio.h>

extern "C" {

// TODO: should have task IDs instead of names
// NOTE: variant ID is one of node IDs in the platform specification
void _crt_mon_task_begin(const char *task_name, uint32_t variant_id)
{
	printf("%s: task %s variant %u\n", __func__, task_name, variant_id);
}

void _crt_mon_task_end(const char *task_name, uint32_t variant_id)
{
	printf("%s: task %s variant %u\n", __func__, task_name, variant_id);
}


} // extern C
