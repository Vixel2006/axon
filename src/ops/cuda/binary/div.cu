#include "ops/cuda/binary.h"
#include "ops/cuda/init.h" // For smalloc, gmalloc (if needed)

__global__ void div_kernel(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = a[i] / (b[i] + 1e-7f);
    }
}

extern "C" void div_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("div_op_cuda: Entering function");
    LOG_INFO("Div kernel starts");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in div_op_cuda");
        assert(0 && "Failed to allocate Storage for out tensor in div_op_cuda");
    }
    out->data->counter = 1;
    out->data->size = N;
    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in div_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        assert(0 && "Failed to allocate CUDA memory for out->data->data in div_op_cuda");
    }

    div_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, b->data->data, out->data->data,
                                                      N);

    CHECK_CUDA();

    LOG_INFO("Div kernel done successfully");
}
