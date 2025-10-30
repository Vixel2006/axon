#ifndef AXON_TENSOR_CPU_H
#define AXON_TENSOR_CPU_H

#include "axon_export.h"
#include "core_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    Storage* smalloc_cpu(float* data, int size, Device* device, Device* src_device);
    void sfree_cpu(Storage* s, Device* device);

#ifdef __cplusplus
}
#endif

#endif // AXON_TENSOR_CPU_H
