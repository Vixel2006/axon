#ifndef AXON_TENSOR_CUDA_H
#define AXON_TENSOR_CUDA_H

#include "axon_export.h"
#include "core_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    Storage* smalloc_cuda(float* data, int size, Device* device);
    void sfree_cuda(Storage* s, Device* device);

#ifdef __cplusplus
}
#endif

#endif // AXON_TENSOR_CUDA_H
