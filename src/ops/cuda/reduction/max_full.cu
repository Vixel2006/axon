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

extern "C" void max_full_op_cuda(Tensor* a, Tensor* out)
{
    LOG_INFO("max operation on cuda starting......");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int blocks = (N + (num_threads_per_block * 2) - 1) / (num_threads_per_block * 2);

    float* h_out_partial_maxs;
    float* d_out_partial_maxs;

    cudaMallocHost((void**) &h_out_partial_maxs, sizeof(float) * blocks);
    cudaMalloc((void**) &d_out_partial_maxs, sizeof(float) * blocks);

    full_max_kernel<256><<<blocks, num_threads_per_block, num_threads_per_block * sizeof(float)>>>(
        a->data->data, d_out_partial_maxs, N);

    cudaMemcpy(h_out_partial_maxs, d_out_partial_maxs, sizeof(float) * blocks,
               cudaMemcpyDeviceToHost);

    float max = h_out_partial_maxs[0];
    for (int i = 0; i < blocks; ++i)
    {
        max = fmaxf(max, h_out_partial_maxs[i]);
    }

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in max_full_op_cuda");
        SAFE_FREE(&h_out_partial_maxs, cudaFreeHost);
        SAFE_FREE(&d_out_partial_maxs, cudaFree);
        return;
    }
    out->data->counter = 1;
    out->data->size = 1; // Scalar output

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in max_full_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        SAFE_FREE(&h_out_partial_maxs, cudaFreeHost);
        SAFE_FREE(&d_out_partial_maxs, cudaFree);
        return;
    }
    err = cudaMemcpy(out->data->data, &max, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to copy final max to CUDA device in max_full_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data->data, cudaFree);
        SAFE_FREE(&out->data, free);
        SAFE_FREE(&h_out_partial_maxs, cudaFreeHost);
        SAFE_FREE(&d_out_partial_maxs, cudaFree);
        return;
    }

    SAFE_FREE(&h_out_partial_maxs, cudaFreeHost);
    SAFE_FREE(&d_out_partial_maxs, cudaFree);

    LOG_INFO("max operation on cuda done.");
}
