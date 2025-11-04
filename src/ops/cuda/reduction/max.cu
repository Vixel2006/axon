#include "ops/cuda/init.h" // For smalloc, gmalloc (if needed)
#include "ops/cuda/reduction.h"
#include "utils/indexing.cuh"

template <int block_size>
__global__ void max_kernel(float* a, float* out, int n, int axis_dim, int inner_dim, int outer_dim)
{
    extern __shared__ float rdata[];
    int tid = threadIdx.x;
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y;

    rdata[tid] = -FLT_MAX;
    __syncthreads();

    for (int i = tid; i < axis_dim; i += block_size)
    {
        int current_index = outer_idx * axis_dim * inner_dim + i * inner_dim + inner_idx;
        if (current_index < n)
        {
            rdata[tid] = fmaxf(rdata[tid], a[current_index]);
        }
    }
    __syncthreads();

    if (block_size >= 512)
    {
        if (tid < 256)
        {
            rdata[tid] = fmaxf(rdata[tid], rdata[tid + 256]);
        }
        __syncthreads();
    }

    if (block_size >= 256)
    {
        if (tid < 128)
        {
            rdata[tid] = fmaxf(rdata[tid], rdata[tid + 128]);
        }
        __syncthreads();
    }

    if (block_size >= 128)
    {
        if (tid < 64)
        {
            rdata[tid] = fmaxf(rdata[tid], rdata[tid + 64]);
        }
        __syncthreads();
    }

    if (block_size >= 64) wrap_max_reduce<block_size>(rdata, tid);

    if (tid == 0)
    {
        int output_idx = outer_idx * inner_dim + inner_idx;
        out[output_idx] = rdata[0];
    }
}

extern "C" void max_op_cuda(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_INFO("max operation on cuda starting......");

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

    if (axis < 0 || axis >= a->ndim)
    {
        LOG_ERROR("max_op_cuda: Axis %d is out of bounds for tensor with %d dimensions.", axis,
                  a->ndim);
        if (a_temp_data) cudaFree(a_temp_data);
        return;
    }

    int outer_dim = 1;
    for (int i = 0; i < axis; ++i)
    {
        outer_dim *= a->shape[i];
    }

    int axis_dim = a->shape[axis];

    int inner_dim = 1;
    for (int i = axis + 1; i < a->ndim; ++i)
    {
        inner_dim *= a->shape[i];
    }

    int num_threads_per_block = 256;
    dim3 grid_dims(outer_dim, inner_dim);

    int output_numel = outer_dim * inner_dim;

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in max_op_cuda");
        if (a_temp_data) cudaFree(a_temp_data);
        return;
    }
    out->data->counter = 1;
    out->data->size = output_numel;

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in max_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        if (a_temp_data) cudaFree(a_temp_data);
        return;
    }

    max_kernel<256><<<grid_dims, num_threads_per_block, num_threads_per_block * sizeof(float)>>>(
        a_data_ptr, out->data->data, N, axis_dim, inner_dim, outer_dim);

    if (a_temp_data)
    {
        cudaFree(a_temp_data);
    }

    CHECK_CUDA();
    LOG_INFO("max operation on cuda done.");
}
