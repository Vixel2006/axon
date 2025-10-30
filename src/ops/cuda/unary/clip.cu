#include "ops/cuda/init.h" // For smalloc, gmalloc (if needed)
#include "ops/cuda/unary.h"

#include "utils/indexing.cuh"

__global__ void noncontig_clip_kernel(const float* a, float* b, const float min_val,
                                      const float max_val, int n, const int* a_shape,
                                      const int* a_strides, int a_ndim, const int* out_shape,
                                      const int* out_strides, int out_ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int a_idx = get_idx(a_shape, a_strides, a_ndim, i);
        int out_idx = get_idx(out_shape, out_strides, out_ndim, i);
        b[out_idx] = fmaxf(min_val, fminf(max_val, a[a_idx]));
    }
}

__global__ void contig_clip_kernel(const float* a, float* b, const float min_val,
                                   const float max_val, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        // NOTE: Here we will use a mix of fminf, fmaxf to metigate the warp diversion problem that
        // will happen if use branching
        b[i] = fmaxf(min_val, fminf(max_val, a[i]));
    }
}

extern "C" void clip_op_cuda(Tensor* in, Tensor* out, float min_val, float max_val)
{
    LOG_INFO("clip_op_cuda: Entering function with min_val=%.2f, max_val=%.2f", min_val, max_val);
    int N = numel(in->shape, in->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in clip_op_cuda");
        assert(0 && "Failed to allocate Storage for out tensor in clip_op_cuda");
    }
    out->data->counter = 1;
    out->data->size = N;

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in clip_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        assert(0 && "Failed to allocate CUDA memory for out->data->data in clip_op_cuda");
    }

    if (is_contiguous(in) && is_contiguous(out))
    {
        contig_clip_kernel<<<num_blocks, num_threads_per_block>>>(in->data->data, out->data->data,
                                                                  min_val, max_val, N);
    }
    else
    {
        noncontig_clip_kernel<<<num_blocks, num_threads_per_block>>>(
            in->data->data, out->data->data, min_val, max_val, N, in->shape, in->strides, in->ndim,
            out->shape, out->strides, out->ndim);
    }

    CHECK_CUDA();
}
