#include "ops/cuda/init.h" // For smalloc, gmalloc (if needed)
#include "ops/cuda/reduction.h"
#include "utils/indexing.cuh"

extern "C" void mean_full_op_cuda(Tensor* a, Tensor* out)
{
    LOG_INFO("mean operation on cuda starting......");

    float* a_data_ptr = a->data->data;
    float* a_temp_data = NULL;
    if (!is_contiguous(a))
    {
        int num_elements = numel(a->shape, a->ndim);
        cudaMalloc((void**) &a_temp_data, num_elements * sizeof(float));
        int num_threads_per_block = 256;
        int num_blocks = (num_elements + num_threads_per_block - 1) / num_threads_per_block;
        copy_non_contiguous_to_contiguous_kernel<<<num_blocks, num_threads_per_block>>>(
            a->data->data, a_temp_data, a->shape, a->strides, a->ndim, num_elements);
        a_data_ptr = a_temp_data;
        CHECK_CUDA();
    }

    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int blocks = (N + (num_threads_per_block * 2) - 1) / (num_threads_per_block * 2);

    float* h_out_partial_sums;
    float* d_out_partial_sums;

    cudaMallocHost((void**) &h_out_partial_sums, sizeof(float) * blocks);
    cudaMalloc((void**) &d_out_partial_sums, sizeof(float) * blocks);

    full_sum_kernel<256><<<blocks, num_threads_per_block, num_threads_per_block * sizeof(float)>>>(
        a_data_ptr, d_out_partial_sums, N);

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
        LOG_ERROR("Failed to allocate Storage for out tensor in mean_full_op_cuda");
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        if (a_temp_data) cudaFree(a_temp_data);
        return;
    }
    out->data->counter = 1;
    out->data->size = 1; // Scalar output

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in mean_full_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        if (a_temp_data) cudaFree(a_temp_data);
        return;
    }
    float final_mean = total_sum / N;
    err = cudaMemcpy(out->data->data, &final_mean, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to copy final mean to CUDA device in mean_full_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data->data, cudaFree);
        SAFE_FREE(&out->data, free);
        SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
        SAFE_FREE(&d_out_partial_sums, cudaFree);
        if (a_temp_data) cudaFree(a_temp_data);
        return;
    }

    SAFE_FREE(&h_out_partial_sums, cudaFreeHost);
    SAFE_FREE(&d_out_partial_sums, cudaFree);
    if (a_temp_data)
    {
        cudaFree(a_temp_data);
    }

    LOG_INFO("mean operation on cuda done.");
}
