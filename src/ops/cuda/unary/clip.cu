#include "ops/cuda/init.h" // For smalloc, gmalloc (if needed)
#include "ops/cuda/unary.h"

__global__ void clip_kernel(const float* a, float* b, const float min_val, const float max_val,
                            int n)
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

    clip_kernel<<<num_blocks, num_threads_per_block>>>(in->data->data, out->data->data, min_val,
                                                       max_val, N);

    CHECK_CUDA();
}
