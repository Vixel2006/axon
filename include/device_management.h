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
    AXON_EXPORT void copy_shape_and_strides_to_device(const int* host_shape, const int* host_strides, int ndim, int** device_shape, int** device_strides);
    AXON_EXPORT void free_device_memory(int* device_shape, int* device_strides);

#ifdef __cplusplus
}
#endif

#endif // AXON_DEVICE_MANAGEMENT_H
