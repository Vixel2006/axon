#include "cuda_utils.h"
#include "utils.h"

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

__global__ void copy_non_contiguous_to_contiguous_kernel(const float* in_data, float* out_data,
                                                         const int* shape, const int* strides,
                                                         int ndim, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_elements; i += stride)
    {
        int in_idx = get_idx(shape, strides, ndim, i);
        out_data[i] = in_data[in_idx];
    }
}
