#include "core_types.h"    // For MAX_NDIM
#include "ops/cuda/init.h" // For smalloc, gmalloc (if needed)
#include "ops/cuda/movement.h"

__global__ void contig_concat_kernel(const float* in_data, float* out_data, size_t outer_size,
                                     size_t in_concat_axis_size, size_t out_concat_axis_size,
                                     size_t inner_size, size_t offset_in_axis)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the total number of elements to process for this input tensor
    size_t total_elements = outer_size * in_concat_axis_size * inner_size;

    if (idx < total_elements)
    {
        // Decompose idx into logical coordinates for the input tensor
        size_t current_inner_idx = idx % inner_size;
        size_t temp = idx / inner_size;
        size_t current_in_concat_axis_idx = temp % in_concat_axis_size;
        size_t current_outer_idx = temp / in_concat_axis_size;

        // Calculate the corresponding index in the output tensor
        size_t out_idx = current_outer_idx * out_concat_axis_size * inner_size +
                         (offset_in_axis + current_in_concat_axis_idx) * inner_size +
                         current_inner_idx;

        out_data[out_idx] = in_data[idx];
    }
}

__global__ void uncontig_concat_kernel(const float* in_data, float* out_data,
                                       const size_t* in_strides, int in_ndim,
                                       const size_t* in_shape, int axis, size_t outer_size,
                                       size_t in_concat_axis_size, size_t out_concat_axis_size,
                                       size_t inner_size, size_t offset_in_axis)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t total_elements = outer_size * in_concat_axis_size * inner_size;

    if (idx < total_elements)
    {
        // Convert linear index 'idx' to multi-dimensional coordinates for the input tensor
        size_t current_coords[MAX_NDIM];
        size_t temp_idx = idx;

        for (int i = in_ndim - 1; i >= 0; --i)
        {
            current_coords[i] = temp_idx % in_shape[i];
            temp_idx /= in_shape[i];
        }

        // Calculate the linear index in the input data based on strides
        size_t in_linear_idx = 0;
        for (int i = 0; i < in_ndim; ++i)
        {
            in_linear_idx += current_coords[i] * in_strides[i];
        }

        // Decompose idx into logical coordinates for the input tensor (similar to contiguous)
        size_t current_inner_idx = idx % inner_size;
        temp_idx = idx / inner_size;
        size_t current_in_concat_axis_idx = temp_idx % in_concat_axis_size;
        size_t current_outer_idx = temp_idx / in_concat_axis_size;

        size_t out_idx = current_outer_idx * out_concat_axis_size * inner_size +
                         (offset_in_axis + current_in_concat_axis_idx) * inner_size +
                         current_inner_idx;

        out_data[out_idx] = in_data[in_linear_idx];
    }
}

extern "C" void concat_op_cuda(Tensor** in, Tensor* out, int num_tensors, int axis)
{
    LOG_INFO("concat_op_cuda: Entering function with num_tensors=%d, axis=%d", num_tensors, axis);

    out->data = (Storage*) malloc(sizeof(Storage));
    assert(out->data && "Failed to allocate Storage for out tensor in concat_op_cuda");
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

    LOG_INFO("concat_op_cuda: Computing the concat of the tensors done successfully.");
}
