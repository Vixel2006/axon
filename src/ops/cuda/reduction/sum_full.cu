#include "ops/cuda/init.h" // For smalloc, gmalloc (if needed)
#include "ops/cuda/reduction.h"

template <int block_size> __global__ void full_sum_kernel(const float* a, float* out, int n)
{
    extern __shared__ float rdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (block_size * 2) + threadIdx.x;
    int grid_size = blockDim.x * 2 * gridDim.x;
    rdata[tid] = 0;

    while (idx < n)
    {
        rdata[tid] += a[idx] + a[idx + block_size];
        idx += grid_size;
    }
    __syncthreads();

    if (block_size >= 512)
    {
        if (tid < 256)
        {
            rdata[tid] += rdata[tid + 256];
        }
        __syncthreads();
    }

    if (block_size >= 256)
    {
        if (tid < 128)
        {
            rdata[tid] += rdata[tid + 128];
        }
        __syncthreads();
    }

    if (block_size >= 128)
    {
        if (tid < 64)
        {
            rdata[tid] += rdata[tid + 64];
        }
        __syncthreads();
    }

    if (block_size >= 64) wrap_sum_reduce<block_size>(rdata, tid);

    if (tid == 0) out[blockIdx.x] = rdata[0];
}

extern "C" void sum_full_op_cuda(Tensor* a, Tensor* out)
{
    LOG_INFO("Sum operation on cuda starting......");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int blocks = (N + (num_threads_per_block * 2) - 1) / (num_threads_per_block * 2);

    float* h_out_partial_sums;
    float* d_out_partial_sums;

    cudaMallocHost((void**) &h_out_partial_sums, sizeof(float) * blocks);
    cudaMalloc((void**) &d_out_partial_sums, sizeof(float) * blocks);

    full_sum_kernel<256><<<blocks, num_threads_per_block, num_threads_per_block * sizeof(float)>>>(
        a->data->data, d_out_partial_sums, N);

    cudaMemcpy(h_out_partial_sums, d_out_partial_sums, sizeof(float) * blocks,
               cudaMemcpyDeviceToHost);

    float total_sum = 0.0f;
    for (int i = 0; i < blocks; ++i)
    {
        total_sum += h_out_partial_sums[i];
    }

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in sum_full_op_cuda");
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        return;
    }
    out->data->counter = 1;
    out->data->size = 1; // Scalar output

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in sum_full_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        return;
    }
    err = cudaMemcpy(out->data->data, &total_sum, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to copy final sum to CUDA device in sum_full_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data->data, cudaFree);
        SAFE_FREE(&out->data, free);
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        return;
    }

    SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
    SAFE_FREE(&d_out_partial_sums, cudaFree);

    LOG_INFO("Sum operation on cuda done.");
}
