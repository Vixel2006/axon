#include "autograd/autograd_movement.h"
#include "logger.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdlib.h>

#define CHECK_CUDA(err)                                                                            \
    do                                                                                             \
    {                                                                                              \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("CUDA runtime error at %s:%d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
            return;                                                                                \
        }                                                                                          \
    } while (0)

#define MAX_DIMS 32

__global__ void concat_grad_kernel_contiguous(const float* out_grad, float* prev_grad,
                                              int outer_size, int prev_concat_axis_size,
                                              int out_concat_axis_size, int inner_size,
                                              int offset_in_axis)
{
    int total_elements = outer_size * prev_concat_axis_size * inner_size;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < total_elements; i += stride)
    {
        int outer_i = i / (prev_concat_axis_size * inner_size);
        int remainder = i % (prev_concat_axis_size * inner_size);

        int out_idx = outer_i * (out_concat_axis_size * inner_size) +
                      (offset_in_axis * inner_size) + remainder;

        prev_grad[i] += out_grad[out_idx];
    }
}

__global__ void concat_grad_kernel_noncontiguous(const float* out_grad, float* prev_grad,
                                                 const int* prev_strides, int prev_ndim,
                                                 const int* prev_shape, int axis, int outer_size,
                                                 int prev_concat_axis_size,
                                                 int out_concat_axis_size, int inner_size,
                                                 int offset_in_axis)
{
    int total_elements = outer_size * prev_concat_axis_size * inner_size;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;

    for (int i = idx; i < total_elements; i += grid_stride)
    {
        if (prev_ndim > MAX_DIMS)
        {
            return;
        }
        int coords[MAX_DIMS];
        int temp_i = i;

        for (int d = prev_ndim - 1; d >= 0; --d)
        {
            coords[d] = temp_i % prev_shape[d];
            temp_i /= prev_shape[d];
        }

        int prev_idx = 0;
        for (int d = 0; d < prev_ndim; ++d)
        {
            prev_idx += coords[d] * prev_strides[d];
        }

        int outer_i = i / (prev_concat_axis_size * inner_size);
        int remainder = i % (prev_concat_axis_size * inner_size);
        int out_idx = outer_i * (out_concat_axis_size * inner_size) +
                      (offset_in_axis * inner_size) + remainder;

        prev_grad[prev_idx] += out_grad[out_idx];
    }
}

void concat_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    ConcatExtras* concat_extras = (ConcatExtras*) extras;
    int axis = concat_extras->axis;
    LOG_INFO("OP: concat_grad_op_cuda: Computing grad for concat of %d tensors around axis %d",
             n_prev, axis);

    if (out->grad == NULL || out->grad->data->data == NULL)
    {
        LOG_WARN("Output tensor has no gradient data, skipping backward pass for concat.");
        return;
    }

    int outer_size = 1;
    for (int i = 0; i < axis; ++i)
    {
        outer_size *= out->shape[i];
    }

    int inner_size = 1;
    for (int i = axis + 1; i < out->ndim; ++i)
    {
        inner_size *= out->shape[i];
    }

    int out_concat_axis_size = out->shape[axis];
    int offset_in_axis = 0;

    for (int i = 0; i < n_prev; ++i)
    {
        Tensor* current_prev = prev[i];

        if (current_prev->requires_grad)
        {
            if (current_prev->grad == NULL || current_prev->grad->data->data == NULL)
            {
                LOG_WARN(
                    "concat_grad_op_cuda: prev tensor %d has no gradient data buffer, skipping.",
                    i);
                offset_in_axis += current_prev->shape[axis];
                continue;
            }
            int N = numel(current_prev->shape, current_prev->ndim);
            int prev_concat_axis_size = current_prev->shape[axis];

            int num_threads_per_block = 256;
            int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

            if (is_contiguous(current_prev))
            {
                concat_grad_kernel_contiguous<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, current_prev->grad->data->data, outer_size,
                    prev_concat_axis_size, out_concat_axis_size, inner_size, offset_in_axis);

                cudaError_t err = cudaGetLastError();
                CHECK_CUDA(err);
            }
            else
            {
                // Handle non-contiguous gradient tensor
                int* d_prev_strides;
                int* d_prev_shape;

                cudaError_t err;
                err = cudaMalloc(&d_prev_strides, current_prev->ndim * sizeof(int));
                if (err != cudaSuccess)
                {
                    LOG_ERROR("Failed to allocate device memory for strides: %s",
                              cudaGetErrorString(err));
                    offset_in_axis += current_prev->shape[axis];
                    continue;
                }

                err = cudaMalloc(&d_prev_shape, current_prev->ndim * sizeof(int));
                if (err != cudaSuccess)
                {
                    LOG_ERROR("Failed to allocate device memory for shape: %s",
                              cudaGetErrorString(err));
                    cudaFree(d_prev_strides);
                    offset_in_axis += current_prev->shape[axis];
                    continue;
                }

                cudaMemcpy(d_prev_strides, current_prev->strides, current_prev->ndim * sizeof(int),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(d_prev_shape, current_prev->shape, current_prev->ndim * sizeof(int),
                           cudaMemcpyHostToDevice);

                concat_grad_kernel_noncontiguous<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, current_prev->grad->data->data, d_prev_strides,
                    current_prev->ndim, d_prev_shape, axis, outer_size, prev_concat_axis_size,
                    out_concat_axis_size, inner_size, offset_in_axis);

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    LOG_ERROR("CUDA kernel launch error at %s:%d: %s", __FILE__, __LINE__,
                              cudaGetErrorString(err));
                    cudaFree(d_prev_strides);
                    cudaFree(d_prev_shape);
                    return;
                }
                cudaFree(d_prev_strides);
                cudaFree(d_prev_shape);
            }
        }

        offset_in_axis += current_prev->shape[axis];
    }

    // Synchronize to ensure all gradients are computed before continuing
    CHECK_CUDA(cudaDeviceSynchronize());

    LOG_INFO("OP: concat_grad_op_cuda: Gradients computed successfully.");
}
