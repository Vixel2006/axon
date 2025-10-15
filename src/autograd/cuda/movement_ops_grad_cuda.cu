#include "autograd/autograd_movement.h"
#include "logger.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdlib.h>

#define CHECK_CUDA()                                                                               \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("CUDA runtime error at %s:%d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
            return;                                                                                \
        }                                                                                          \
    } while (0)

__global__ void concat_grad_kernel_contiguous(const float* out_grad, float* prev_grad,
                                              size_t outer_size, size_t prev_concat_axis_size,
                                              size_t out_concat_axis_size, size_t inner_size,
                                              size_t offset_in_axis)
{
    size_t total_elements = outer_size * prev_concat_axis_size * inner_size;
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < total_elements; i += stride)
    {
        size_t outer_i = i / (prev_concat_axis_size * inner_size);
        size_t remainder = i % (prev_concat_axis_size * inner_size);

        size_t out_idx = outer_i * (out_concat_axis_size * inner_size) +
                         (offset_in_axis * inner_size) + remainder;

        prev_grad[i] += out_grad[out_idx];
    }
}

__global__ void concat_grad_kernel_noncontiguous(const float* out_grad, float* prev_grad,
                                                 const size_t* prev_strides, int prev_ndim,
                                                 const size_t* prev_shape, int axis,
                                                 size_t outer_size, size_t prev_concat_axis_size,
                                                 size_t out_concat_axis_size, size_t inner_size,
                                                 size_t offset_in_axis)
{
    size_t total_elements = outer_size * prev_concat_axis_size * inner_size;
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t grid_stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < total_elements; i += grid_stride)
    {
        size_t coords[10];
        size_t temp_i = i;

        // Decompose linear index into coordinates
        for (int d = prev_ndim - 1; d > axis; --d)
        {
            coords[d] = temp_i % prev_shape[d];
            temp_i /= prev_shape[d];
        }

        coords[axis] = temp_i % prev_shape[axis];
        temp_i /= prev_shape[axis];

        for (int d = axis - 1; d >= 0; --d)
        {
            coords[d] = temp_i % prev_shape[d];
            temp_i /= prev_shape[d];
        }

        // Calculate destination index in prev_grad using strides
        size_t prev_idx = 0;
        for (int d = 0; d < prev_ndim; ++d)
        {
            prev_idx += coords[d] * prev_strides[d];
        }

        // Calculate source index in out_grad (always contiguous)
        size_t outer_i = i / (prev_concat_axis_size * inner_size);
        size_t remainder = i % (prev_concat_axis_size * inner_size);
        size_t out_idx = outer_i * (out_concat_axis_size * inner_size) +
                         (offset_in_axis * inner_size) + remainder;

        // Use atomicAdd for thread-safe accumulation
        atomicAdd(&prev_grad[prev_idx], out_grad[out_idx]);
    }
}

/*
 * Be a stubbern ass bro.
 * You need to finish the fucking concat grad op on cuda kernels
 * what we need to do is for fuck sake only map output index to output index
 * just like that you stupid piece of shit. fuck you
 * hwo to do this bozzoo.
 * we can just find a way to map the two shitty stuff. we need examples 2 inputs (2,2,2) and 1
 * output (4,2,2) that if we concat around axis=0 we need to solve this right now. index 0, tensor 0
 * => (0,0,0) and (0,0,0) in both which makes me think that this case is very trivial let's think of
 * the possibility of (2,4,2)  what we do here is literally [0, 0, :] of the first of all tensors =
 * we can just go for the [0,0,:], and [0,1,:] from the first one, [0,2,:] and [0,3,:] from the
 * second so what we should do I really think I'm very close to solve this
 * */

void concat_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    int axis = *(int*) extras;
    LOG_INFO("OP: concat_grad_op_cuda: Computing grad for concat of %d tensors around axis %d",
             n_prev, axis);

    if (out->grad == NULL || out->grad->data == NULL)
    {
        LOG_WARN("Output tensor has no gradient data, skipping backward pass for concat.");
        return;
    }

    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i)
    {
        outer_size *= out->shape[i];
    }

    size_t inner_size = 1;
    for (int i = axis + 1; i < out->ndim; ++i)
    {
        inner_size *= out->shape[i];
    }

    size_t out_concat_axis_size = out->shape[axis];
    size_t offset_in_axis = 0;

    for (int i = 0; i < n_prev; ++i)
    {
        Tensor* current_prev = prev[i];

        if (current_prev->requires_grad)
        {
            size_t N = numel(current_prev->shape, current_prev->ndim);
            size_t prev_concat_axis_size = current_prev->shape[axis];

            int num_threads_per_block = 256;
            int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

            if (is_contiguous(current_prev))
            {
                concat_grad_kernel_contiguous<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data, current_prev->grad->data, outer_size, prev_concat_axis_size,
                    out_concat_axis_size, inner_size, offset_in_axis);

                CHECK_CUDA();
            }
            else
            {
                // Handle non-contiguous gradient tensor
                size_t* d_prev_strides;
                size_t* d_prev_shape;

                cudaError_t err;
                err = cudaMalloc(&d_prev_strides, current_prev->ndim * sizeof(size_t));
                if (err != cudaSuccess)
                {
                    LOG_ERROR("Failed to allocate device memory for strides: %s",
                              cudaGetErrorString(err));
                    offset_in_axis += current_prev->shape[axis];
                    continue;
                }

                err = cudaMalloc(&d_prev_shape, current_prev->ndim * sizeof(size_t));
                if (err != cudaSuccess)
                {
                    LOG_ERROR("Failed to allocate device memory for shape: %s",
                              cudaGetErrorString(err));
                    cudaFree(d_prev_strides);
                    offset_in_axis += current_prev->shape[axis];
                    continue;
                }

                cudaMemcpy(d_prev_strides, current_prev->strides,
                           current_prev->ndim * sizeof(size_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_prev_shape, current_prev->shape, current_prev->ndim * sizeof(size_t),
                           cudaMemcpyHostToDevice);

                concat_grad_kernel_noncontiguous<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data, current_prev->grad->data, d_prev_strides, current_prev->ndim,
                    d_prev_shape, axis, outer_size, prev_concat_axis_size, out_concat_axis_size,
                    inner_size, offset_in_axis);

                CHECK_CUDA();
                cudaFree(d_prev_strides);
                cudaFree(d_prev_shape);
            }
        }

        offset_in_axis += current_prev->shape[axis];
    }

    // Synchronize to ensure all gradients are computed before continuing
    cudaDeviceSynchronize();
    CHECK_CUDA();

    LOG_INFO("OP: concat_grad_op_cuda: Gradients computed successfully.");
}
