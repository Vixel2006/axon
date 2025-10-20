#ifndef AXON_DEVICE_MANAGEMENT_H
#define AXON_DEVICE_MANAGEMENT_H

#include "axon_export.h"

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT int count_devices();
    AXON_EXPORT bool is_cuda_available();
    AXON_EXPORT void print_device_props();
    AXON_EXPORT void print_cuda_device_info(int index);
    AXON_EXPORT void get_cuda_memory_info(int device_id);

#ifdef __cplusplus
}
#endif

#endif // AXON_DEVICE_MANAGEMENT_H
