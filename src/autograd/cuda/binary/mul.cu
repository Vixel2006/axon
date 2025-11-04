#include "autograd/cuda/binary/common.cuh"
#include "autograd/cuda/broadcast_utils.cuh"
#include "utils/indexing.cuh"
#include "device_management.h" // Added for device memory management

// Helper functions for comparing shapes and strides
__host__ __device__ bool compare_shapes(const int* shape1, int ndim1, const int* shape2, int ndim2) {
    if (ndim1 != ndim2) {
        return false;
    }
    for (int i = 0; i < ndim1; ++i) {
        if (shape1[i] != shape2[i]) {
            return false;
        }
    }
    return true;
}

__host__ __device__ bool compare_strides(const int* strides1, int ndim1, const int* strides2, int ndim2) {
    if (ndim1 != ndim2) {
        return false;
    }
    for (int i = 0; i < ndim1; ++i) {
        if (strides1[i] != strides2[i]) {
            return false;
        }
    }
    return true;
}

__global__ void mul_grad_kernel(const float* out_grad, float* prev_grad, float* other_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] * other_data[i];
    }
}

__global__ void scalar_mul_grad_kernel(const float* out_grad, float* prev_grad, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] * scalar;
    }
}

__global__ void noncontig_mul_grad_kernel(const float* out_grad, float* prev_grad,
                                          const float* other_data, int n, const int* prev_shape,
                                          const int* prev_strides, int prev_ndim,
                                          const int* other_shape, const int* other_strides,
                                          int other_ndim, const int* out_shape,
                                          const int* out_strides, int out_ndim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int prev_in_idx =
            get_broadcasted_input_idx(i, out_shape, out_ndim, prev_shape, prev_strides, prev_ndim);
        int other_in_idx = get_broadcasted_input_idx(i, out_shape, out_ndim, other_shape,
                                                     other_strides, other_ndim);

        atomicAdd(&prev_grad[prev_in_idx], out_grad[i] * other_data[other_in_idx]);
    }
}

void mul_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("mul_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert((n_prev == 1 || n_prev == 2) && "n_prev must be 1 or 2 for mul_grad_op_cuda");

    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1)
    {
        assert(extras && "Extras (scalar value) cannot be NULL for scalar multiplication");
        float* scalar = (float*) extras;
        assert(prev[0] && "Previous tensor 0 cannot be NULL");
        assert(prev[0]->data && "Previous tensor 0 data cannot be NULL");
        assert(prev[0]->data->data && "Previous tensor 0 data pointer cannot be NULL");
        if (prev[0]->requires_grad)
        {
            assert(prev[0]->grad && "Previous tensor 0 gradient cannot be NULL if requires_grad");
            assert(prev[0]->grad->data &&
                   "Previous tensor 0 gradient data cannot be NULL if requires_grad");
            assert(prev[0]->grad->data->data &&
                   "Previous tensor 0 gradient data pointer cannot be NULL if requires_grad");
            scalar_mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, prev[0]->grad->data->data, *scalar, N);
            CHECK_CUDA();
        }
    }
    else if (n_prev == 2)
    {
        assert(prev[0] && "Previous tensor 0 cannot be NULL");
        assert(prev[1] && "Previous tensor 1 cannot be NULL");

        // Determine if broadcasting is needed
        bool prev0_broadcasted = !compare_shapes(prev[0]->shape, prev[0]->ndim, out->shape, out->ndim) ||
                                 !compare_strides(prev[0]->strides, prev[0]->ndim, out->strides, out->ndim);
        bool prev1_broadcasted = !compare_shapes(prev[1]->shape, prev[1]->ndim, out->shape, out->ndim) ||
                                 !compare_strides(prev[1]->strides, prev[1]->ndim, out->strides, out->ndim);

        if (!prev0_broadcasted && !prev1_broadcasted)
        {
            // No broadcasting, use simple mul_grad_kernel
            if (prev[0]->requires_grad)
            {
                mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->grad->data->data, prev[1]->data->data, N);
                CHECK_CUDA();
            }
            if (prev[1]->requires_grad)
            {
                mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[1]->grad->data->data, prev[0]->data->data, N);
                CHECK_CUDA();
            }
        }
        else
        {
            // Broadcasting is involved, use noncontig_mul_grad_kernel
            int *d_out_shape, *d_out_strides;
            int *d_prev0_shape, *d_prev0_strides;
            int *d_prev1_shape, *d_prev1_strides;

            copy_shape_and_strides_to_device(out->shape, out->strides, out->ndim, &d_out_shape, &d_out_strides);
            copy_shape_and_strides_to_device(prev[0]->shape, prev[0]->strides, prev[0]->ndim, &d_prev0_shape, &d_prev0_strides);
            copy_shape_and_strides_to_device(prev[1]->shape, prev[1]->strides, prev[1]->ndim, &d_prev1_shape, &d_prev1_strides);

            if (prev[0]->requires_grad)
            {
                noncontig_mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[0]->grad->data->data, prev[1]->data->data, N,
                    d_prev0_shape, d_prev0_strides, prev[0]->ndim,
                    d_prev1_shape, d_prev1_strides, prev[1]->ndim,
                    d_out_shape, d_out_strides, out->ndim);
                CHECK_CUDA();
            }
            if (prev[1]->requires_grad)
            {
                noncontig_mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, prev[1]->grad->data->data, prev[0]->data->data, N,
                    d_prev1_shape, d_prev1_strides, prev[1]->ndim,
                    d_prev0_shape, d_prev0_strides, prev[0]->ndim,
                    d_out_shape, d_out_strides, out->ndim);
                CHECK_CUDA();
            }

            free_device_memory(d_out_shape, d_out_strides);
            free_device_memory(d_prev0_shape, d_prev0_strides);
            free_device_memory(d_prev1_shape, d_prev1_strides);
        }
    }
}
