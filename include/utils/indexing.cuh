#ifndef AXON_CUDA_INDEXING_UTIL
#define AXON_CUDA_INDEXING_UTIL

#include <cuda_runtime.h>

__device__ __forceinline__ int get_idx(const int* shape, const int* strides, int ndim, int i)
{
    int tmp = i;
    int data_idx = 0;
    for (int d = ndim - 1; d >= 0; --d)
    {
        int dim_idx = tmp % shape[d];
        data_idx += dim_idx * strides[d];
        tmp /= shape[d];
    }

    return data_idx;
}

#endif
