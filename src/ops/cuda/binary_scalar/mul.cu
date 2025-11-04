#include "ops/cuda/binary_scalar.h"
#include "utils/indexing.cuh"
#include <assert.h>

__global__ void noncontig_mul_scalar_kernel(const float* in, float* out, float scalar, int n,
                                            const int* in_shape, const int* in_strides, int in_ndim,
                                            const int* out_shape, const int* out_strides,
                                            int out_ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx = get_idx(in_shape, in_strides, in_ndim, i);
        int out_idx = get_idx(out_shape, out_strides, out_ndim, i);

        out[out_idx] = in[in_idx] * scalar;
    }
}

__global__ void contig_mul_scalar_kernel(const float* in, float* out, float scalar, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = in[i] * scalar;
    }
}

void mul_scalar_op_cuda(Tensor* a, float b, Tensor* out)
{
    LOG_INFO("mul_scalar_op_cuda: Entering function");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in mul_scalar_op_cuda");
        assert(0 && "Failed to allocate Storage for out tensor in mul_scalar_op_cuda");
    }
    out->data->counter = 1;
    out->data->size = N;

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in mul_scalar_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        assert(0 && "Failed to allocate CUDA memory for out->data->data in mul_scalar_op_cuda");
    }

    if (is_contiguous(a) && is_contiguous(out))
    {
        contig_mul_scalar_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, out->data->data, b, N);
    }
    else
    {
        noncontig_mul_scalar_kernel<<<num_blocks, num_threads_per_block>>>(
            a->data->data, out->data->data, b, N, a->shape, a->strides, a->ndim, out->shape,
            out->strides, out->ndim);
    }

    CHECK_CUDA();

    LOG_INFO("mul_scalar_op_cuda: done successfully");
}
