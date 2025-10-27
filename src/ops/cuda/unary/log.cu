#include "ops/cuda/init.h" // For smalloc, gmalloc (if needed)
#include "ops/cuda/unary.h"

__global__ void log_kernel(const float* a, float* b, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        b[i] = logf(a[i] + 1e-7f);
    }
}

extern "C" void log_op_cuda(Tensor* in, Tensor* out)
{
    LOG_INFO("log_op_cuda: Entering function");
    int N = numel(in->shape, in->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in log_op_cuda");
        assert(0 && "Failed to allocate Storage for out tensor in log_op_cuda");
    }
    out->data->counter = 1;
    out->data->size = N;

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in log_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        assert(0 && "Failed to allocate CUDA memory for out->data->data in log_op_cuda");
    }

    log_kernel<<<num_blocks, num_threads_per_block>>>(in->data->data, out->data->data, N);

    CHECK_CUDA();
}
