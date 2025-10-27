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
    void copy_non_contiguous_cuda_to_host_kernel(float* d_data, const int* shape,
                                                 const int* strides, int ndim, int num_elements,
                                                 float* h_buffer);
    void copy_non_contiguous_cuda_to_host(Storage* s, const int* shape, const int* strides,
                                          int ndim, int num_elements, float* host_buffer);

#ifdef __cplusplus
}
#endif

#endif // AXON_TENSOR_CUDA_H
