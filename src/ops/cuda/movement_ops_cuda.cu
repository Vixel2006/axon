#include "logger.h"
#include "ops/movement_ops.h"
#include <cuda_runtime.h>

#define CHECK_CUDA(err)                                                                            \
    do                                                                                             \
    {                                                                                              \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("cuda-runtime error at %s %d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
        }                                                                                          \
    } while (0)

__global__ void contig_concat_kernel(const float* in_data, float* out_data, size_t outer_size,
                                     size_t in_concat_axis_size, size_t out_concat_axis_size,
                                     size_t inner_size, size_t offset_in_axis)
{
    size_t total_elements = outer_size * in_concat_axis_size * inner_size;
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < total_elements; i += stride)
    {
        size_t outer_i = i / (in_concat_axis_size * inner_size);
        size_t remainder = i % (in_concat_axis_size * inner_size);

        size_t out_idx = outer_i * (out_concat_axis_size * inner_size) +
                         (offset_in_axis * inner_size) + remainder;

        out_data[out_idx] = in_data[i];
    }
}

__global__ void uncontig_concat_kernel(const float* in_data, float* out_data,
                                       const size_t* in_strides, int in_ndim,
                                       const size_t* in_shape, int axis, size_t outer_size,
                                       size_t in_concat_axis_size, size_t out_concat_axis_size,
                                       size_t inner_size, size_t offset_in_axis)
{
    size_t total_elements = outer_size * in_concat_axis_size * inner_size;
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t grid_stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < total_elements; i += grid_stride)
    {
        size_t coords[10];
        size_t temp_i = i;

        for (int d = in_ndim - 1; d > axis; --d)
        {
            coords[d] = temp_i % in_shape[d];
            temp_i /= in_shape[d];
        }

        coords[axis] = temp_i % in_shape[axis];
        temp_i /= in_shape[axis];

        for (int d = axis - 1; d >= 0; --d)
        {
            coords[d] = temp_i % in_shape[d];
            temp_i /= in_shape[d];
        }

        // Calculate source index using strides
        size_t in_idx = 0;
        for (int d = 0; d < in_ndim; ++d)
        {
            in_idx += coords[d] * in_strides[d];
        }

        // Calculate destination index
        size_t outer_i = i / (in_concat_axis_size * inner_size);
        size_t remainder = i % (in_concat_axis_size * inner_size);
        size_t out_idx = outer_i * (out_concat_axis_size * inner_size) +
                         (offset_in_axis * inner_size) + remainder;

        out_data[out_idx] = in_data[in_idx];
    }
}

void concat_op_cuda(Tensor** in, Tensor* out, int num_tensors, int axis)
{
    LOG_INFO("OP: concat_op_cuda: Computing the concat of the %d tensors around axis %d",
             num_tensors, axis);
    out->data = (Storage*) malloc(sizeof(Storage));
    out->data->counter = 1;
    out->data->size = numel(out->shape, out->ndim);
    CHECK_CUDA(cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float)));

    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i)
    {
        outer_size *= out->shape[i];
    }

    size_t out_concat_axis_size = out->shape[axis];

    size_t inner_size = 1;
    for (int i = axis + 1; i < out->ndim; ++i)
    {
        inner_size *= out->shape[i];
    }

    size_t offset_in_axis = 0;
    for (int idx = 0; idx < num_tensors; ++idx)
    {
        Tensor* current_in = in[idx];
        if (is_contiguous(current_in))
        {
            size_t N = numel(current_in->shape, current_in->ndim);
            size_t in_concat_axis_size = current_in->shape[axis];

            int num_threads_per_block = 256;
            int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

            contig_concat_kernel<<<num_blocks, num_threads_per_block>>>(
                current_in->data->data, out->data->data, outer_size, in_concat_axis_size,
                out_concat_axis_size, inner_size, offset_in_axis);

            CHECK_CUDA(cudaGetLastError());
        }
        else
        {
            size_t N = numel(current_in->shape, current_in->ndim);
            size_t in_concat_axis_size = current_in->shape[axis];

            int num_threads_per_block = 256;
            int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

            size_t* d_in_strides;
            size_t* d_in_shape;
            CHECK_CUDA(cudaMalloc(&d_in_strides, current_in->ndim * sizeof(size_t)));
            CHECK_CUDA(cudaMalloc(&d_in_shape, current_in->ndim * sizeof(size_t)));
            CHECK_CUDA(cudaMemcpy(d_in_strides, current_in->strides,
                                  current_in->ndim * sizeof(size_t), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_in_shape, current_in->shape, current_in->ndim * sizeof(size_t),
                                  cudaMemcpyHostToDevice));

            uncontig_concat_kernel<<<num_blocks, num_threads_per_block>>>(
                current_in->data->data, out->data->data, d_in_strides, current_in->ndim, d_in_shape,
                axis, outer_size, in_concat_axis_size, out_concat_axis_size, inner_size,
                offset_in_axis);

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaFree(d_in_strides));
            CHECK_CUDA(cudaFree(d_in_shape));
        }
        offset_in_axis += current_in->shape[axis];
    }

    LOG_INFO("OP: concat_op_cuda: Computing the concat of the tensors done successfully.");
}
