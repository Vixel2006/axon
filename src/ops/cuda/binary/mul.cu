#include "ops/cuda/binary.h"
#include "ops/cuda/init.h" // For smalloc, gmalloc (if needed)

#include "utils/indexing.cuh"

__global__ void noncontig_mul_kernel(const float* a, const float* b, float* out, const int n,
                                     const int* a_shape, const int* a_strides, int a_ndim,
                                     const int* b_shape, const int* b_strides, int b_ndim,
                                     const int* out_shape, const int* out_strides, int out_ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int a_idx = get_idx(a_shape, a_strides, a_ndim, i);
        int b_idx = get_idx(b_shape, b_strides, b_ndim, i);
        int out_idx = get_idx(out_shape, out_strides, out_ndim, i);

        out[out_idx] = a[a_idx] * b[b_idx];
    }
}

__global__ void contig_mul_kernel(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = a[i] * b[i];
    }
}

extern "C" void mul_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("mul_op_cuda: Entering function");
    LOG_INFO("Mul kernel starts");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in mul_op_cuda");
        assert(0 && "Failed to allocate Storage for out tensor in mul_op_cuda");
    }
    out->data->counter = 1;
    out->data->size = N;
    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in mul_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        assert(0 && "Failed to allocate CUDA memory for out->data->data in mul_op_cuda");
    }

    if (is_contiguous(a) && is_contiguous(b) && is_contiguous(out))
    {
        contig_mul_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, b->data->data,
                                                                 out->data->data, N);
    }
    else
    {
        noncontig_mul_kernel<<<num_blocks, num_threads_per_block>>>(
            a->data->data, b->data->data, out->data->data, N, a->shape, a->strides, a->ndim,
            b->shape, b->strides, b->ndim, out->shape, out->strides, out->ndim);
    }
    CHECK_CUDA();

    LOG_INFO("Mul kernel done successfully");
}
