#include "autograd/cuda/reduction/common.cuh"
#include "autograd/cuda/reduction/reduction_ops_cuda.h"
#include "ops/cuda/init.h" // For smalloc, gmalloc (if needed)
#include "ops/cuda/reduction.h"
#include <float.h>

template <int block_size> __global__ void full_max_kernel(float* a, float* out, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (block_size * 2) + tid;
    unsigned int gridSize = block_size * 2 * gridDim.x;

    sdata[tid] = -FLT_MAX; // Initialize with smallest possible float

    while (i < n)
    {
        sdata[tid] = fmaxf(sdata[tid], a[i]);
        if (i + block_size < n)
        {
            sdata[tid] = fmaxf(sdata[tid], a[i + block_size]);
        }
        i += gridSize;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = block_size / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = sdata[0];
    }
}

__global__ void max_full_grad_kernel(float* in_grad_data, float* in_data, float* output_grad,
                                     int in_size, float* max, const int* in_grad_shape,
                                     const int* in_grad_strides, int in_grad_ndim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < in_size; i += stride)
    {
        int in_grad_idx = get_idx(in_grad_shape, in_grad_strides, in_grad_ndim, i);
        // Use an epsilon-based comparison for floating-point equality
        if (fabsf(in_data[i] - max[0]) < EPSILON)
        {
            in_grad_data[in_grad_idx] += output_grad[0];
        }
    }
}

void max_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("max_full_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(out->data && "Output tensor data cannot be NULL");
    assert(out->data->data && "Output tensor data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 1 && "n_prev must be 1 for max_full_grad_op_cuda");

    Tensor* a = prev[0];
    assert(a && "Input tensor 'a' cannot be NULL");
    assert(a->data && "Input tensor 'a' data cannot be NULL");
    assert(a->data->data && "Input tensor 'a' data pointer cannot be NULL");

    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (a->requires_grad)
    {
        assert(a->grad && "Input tensor 'a' gradient cannot be NULL if requires_grad");
        assert(a->grad->data && "Input tensor 'a' gradient data cannot be NULL if requires_grad");
        assert(a->grad->data->data &&
               "Input tensor 'a' gradient data pointer cannot be NULL if requires_grad");
        max_full_grad_kernel<<<num_blocks, num_threads_per_block>>>(
            a->grad->data->data, a->data->data, out->grad->data->data, N, out->data->data, a->shape,
            a->strides, a->ndim);

        CHECK_CUDA();
    }
}
