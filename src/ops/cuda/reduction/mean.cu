#include "ops/cuda/init.h" // For smalloc, gmalloc (if needed)
#include "ops/cuda/reduction.h"

template <int block_size>
__global__ void mean_kernel(float* a, float* out, int n, int axis_dim, int outer_dim, int inner_dim)
{
    extern __shared__ float rdata[];
    int tid = threadIdx.x;
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y;

    rdata[tid] = 0.0f;
    __syncthreads();

    for (int i = tid; i < axis_dim; i += 2 * block_size)
    {
        int left_index = outer_idx * axis_dim * inner_dim + i * inner_dim + inner_idx;
        int right_index =
            outer_idx * axis_dim * inner_dim + (i + block_size) * inner_dim + inner_idx;
        float left_val = left_index < n ? a[left_index] : 0.0f;
        float right_val = right_index < n ? a[right_index] : 0.0f;
        rdata[tid] += left_val + right_val;
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

    if (tid == 0)
    {
        int output_idx = outer_idx * inner_dim + inner_idx;
        out[output_idx] = rdata[0] / axis_dim;
    }
}

extern "C" void mean_op_cuda(Tensor* a, Tensor* out, int axis, bool keepdim)
{
    LOG_INFO("mean operation on cuda starting......");

    int N = numel(a->shape, a->ndim);

    if (axis < 0 || axis >= a->ndim)
    {
        LOG_ERROR("mean_op_cuda: Axis %d is out of bounds for tensor with %d dimensions.", axis,
                  a->ndim);
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
        LOG_ERROR("Failed to allocate Storage for out tensor in mean_op_cuda");
        return;
    }
    out->data->counter = 1;
    out->data->size = output_numel;

    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in mean_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        return;
    }

    mean_kernel<256><<<grid_dims, num_threads_per_block, num_threads_per_block * sizeof(float)>>>(
        a->data->data, out->data->data, N, axis_dim, outer_dim, inner_dim);

    CHECK_CUDA();
    LOG_INFO("mean operation on cuda done.");
}
