#include "autograd/autograd_binary.h"
#include "logger.h"
#include "ops/movement_ops.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_DIM 16

#define CHECK_CUDA()                                                                               \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("CUDA runtime error at %s:%d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
            assert(0 && "CUDA runtime error");                                                     \
        }                                                                                          \
    } while (0)

__global__ void add_grad_kernel(const float* out_grad, float* prev_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i];
    }
}

__global__ void sub_grad_kernel(const float* out_grad, float* prev_grad, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] -= out_grad[i];
    }
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

__global__ void scalar_pow_grad_kernel(const float* out_grad, float* prev_data, float* prev_grad,
                                       float power, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += power * powf(prev_data[i] + 1e-7f, power - 1) * out_grad[i];
    }
}

__global__ void base_pow_grad_kernel(const float* out_grad, float* base_data, float* base_grad,
                                     float* power_data, float* power_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        base_grad[i] += power_data[i] * powf(base_data[i] + 1e-7f, power_data[i] - 1) * out_grad[i];
    }
}

__global__ void exponent_pow_grad_kernel(const float* out_grad, const float* out_data,
                                         float* base_data, float* power_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        power_grad[i] += out_data[i] * logf(base_data[i] + 1e-7f);
    }
}

__global__ void numerator_div_grad_kernel(const float* out_grad, float* prev_grad,
                                          const float* denominator, int n)
{
    int idx = blockDim.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] / (denominator[i] + 1e-7f);
    }
}

__global__ void denominator_div_grad_kernel(const float* out_grad, const float* out_data,
                                            float* prev_grad, float* denominator, int n)
{
    int idx = blockDim.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] -= out_data[i] * out_grad[i] / (denominator[i] + 1e-7f);
    }
}

__global__ void scalar_div_grad_kernel(const float* out_grad, float* prev_grad,
                                       float scalar_denominator, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] / (scalar_denominator + 1e-7f);
    }
}

__global__ void scalar_rdiv_grad_kernel(const float* out_grad, const float* out_data,
                                        float* prev_grad, float scalar_numerator,
                                        const float* prev_data, int n)
{
    int idx = blockDim.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] -= out_grad[i] * out_data[i] / (prev_data[i] + 1e-7f);
    }
}

__global__ void matmul_grad_kernel(const float* lhs, const float* rhs, float* grad, int B, int N,
                                   int P, int K, bool transpose_lhs, bool transpose_rhs,
                                   bool is_lhs_batched, bool is_rhs_batched, bool is_grad_batched)
{
    __shared__ float lhs_tile[TILE_DIM][TILE_DIM];
    __shared__ float rhs_tile[TILE_DIM][TILE_DIM];

    int batch = blockIdx.z;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;

    int lhs_storage_rows = transpose_lhs ? K : N;
    int lhs_storage_cols = transpose_lhs ? N : K;
    int rhs_storage_rows = transpose_rhs ? P : K;
    int rhs_storage_cols = transpose_rhs ? K : P;

    const float* batched_lhs =
        is_lhs_batched ? lhs + batch * lhs_storage_rows * lhs_storage_cols : lhs;
    const float* batched_rhs =
        is_rhs_batched ? rhs + batch * rhs_storage_rows * rhs_storage_cols : rhs;
    float* grad_ptr = is_grad_batched ? grad + batch * N * P : grad;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        int inner_dim_idx_lhs = TILE_DIM * t + threadIdx.x;
        int inner_dim_idx_rhs = TILE_DIM * t + threadIdx.y;

        if (row < N && inner_dim_idx_lhs < K)
        {
            if (transpose_lhs)
            {
                lhs_tile[threadIdx.y][threadIdx.x] = batched_lhs[inner_dim_idx_lhs * N + row];
            }
            else
            {
                lhs_tile[threadIdx.y][threadIdx.x] = batched_lhs[row * K + inner_dim_idx_lhs];
            }
        }
        else
        {
            lhs_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (inner_dim_idx_rhs < K && col < P)
        {
            if (transpose_rhs)
            {
                rhs_tile[threadIdx.y][threadIdx.x] = batched_rhs[col * K + inner_dim_idx_rhs];
            }
            else
            {
                rhs_tile[threadIdx.y][threadIdx.x] = batched_rhs[inner_dim_idx_rhs * P + col];
            }
        }
        else
        {
            rhs_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k_tile = 0; k_tile < TILE_DIM; ++k_tile)
        {
            sum += lhs_tile[threadIdx.y][k_tile] * rhs_tile[k_tile][threadIdx.x];
        }
        __syncthreads();
    }

    if (col < P && row < N)
    {
        if (is_grad_batched)
        {
            grad_ptr[row * P + col] += sum;
        }
        else
        {
            atomicAdd(&grad_ptr[row * P + col], sum);
        }
    }
}

void add_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("add_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert((n_prev == 1 || n_prev == 2) && "n_prev must be 1 or 2 for add_grad_op_cuda");

    int N = numel(out->shape, out->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1)
    {
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
            add_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data->data,
                                                                   prev[0]->grad->data->data, N);
            CHECK_CUDA();
        }
    }
    else
    {
        assert(prev[0] && "Previous tensor 0 cannot be NULL");
        assert(prev[0]->data && "Previous tensor 0 data cannot be NULL");
        assert(prev[0]->data->data && "Previous tensor 0 data pointer cannot be NULL");
        assert(prev[1] && "Previous tensor 1 cannot be NULL");
        assert(prev[1]->data && "Previous tensor 1 data cannot be NULL");
        assert(prev[1]->data->data && "Previous tensor 1 data pointer cannot be NULL");

        if (prev[0]->requires_grad)
        {
            assert(prev[0]->grad && "Previous tensor 0 gradient cannot be NULL if requires_grad");
            assert(prev[0]->grad->data &&
                   "Previous tensor 0 gradient data cannot be NULL if requires_grad");
            assert(prev[0]->grad->data->data &&
                   "Previous tensor 0 gradient data pointer cannot be NULL if requires_grad");
            add_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data->data,
                                                                   prev[0]->grad->data->data, N);
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad)
        {
            assert(prev[1]->grad && "Previous tensor 1 gradient cannot be NULL if requires_grad");
            assert(prev[1]->grad->data &&
                   "Previous tensor 1 gradient data cannot be NULL if requires_grad");
            assert(prev[1]->grad->data->data &&
                   "Previous tensor 1 gradient data pointer cannot be NULL if requires_grad");
            add_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data->data,
                                                                   prev[1]->grad->data->data, N);
            CHECK_CUDA();
        }
    }
}

void sub_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("sub_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert((n_prev == 1 || n_prev == 2) && "n_prev must be 1 or 2 for sub_grad_op_cuda");

    int N = numel(out->shape, out->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1)
    {
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
            add_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data->data,
                                                                   prev[0]->grad->data->data, N);
            CHECK_CUDA();
        }
    }
    else
    {
        assert(prev[0] && "Previous tensor 0 cannot be NULL");
        assert(prev[0]->data && "Previous tensor 0 data cannot be NULL");
        assert(prev[0]->data->data && "Previous tensor 0 data pointer cannot be NULL");
        assert(prev[1] && "Previous tensor 1 cannot be NULL");
        assert(prev[1]->data && "Previous tensor 1 data cannot be NULL");
        assert(prev[1]->data->data && "Previous tensor 1 data pointer cannot be NULL");

        if (prev[0]->requires_grad)
        {
            assert(prev[0]->grad && "Previous tensor 0 gradient cannot be NULL if requires_grad");
            assert(prev[0]->grad->data &&
                   "Previous tensor 0 gradient data cannot be NULL if requires_grad");
            assert(prev[0]->grad->data->data &&
                   "Previous tensor 0 gradient data pointer cannot be NULL if requires_grad");
            add_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data->data,
                                                                   prev[0]->grad->data->data, N);
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad)
        {
            assert(prev[1]->grad && "Previous tensor 1 gradient cannot be NULL if requires_grad");
            assert(prev[1]->grad->data &&
                   "Previous tensor 1 gradient data cannot be NULL if requires_grad");
            assert(prev[1]->grad->data->data &&
                   "Previous tensor 1 gradient data pointer cannot be NULL if requires_grad");
            sub_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data->data,
                                                                   prev[1]->grad->data->data, N);
            CHECK_CUDA();
        }
    }
}

void rsub_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("rsub_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 1 && "n_prev must be 1 for rsub_grad_op_cuda");
    assert(prev[0] && "Previous tensor 0 cannot be NULL");
    assert(prev[0]->data && "Previous tensor 0 data cannot be NULL");
    assert(prev[0]->data->data && "Previous tensor 0 data pointer cannot be NULL");
    assert(extras && "Extras (scalar value) cannot be NULL");

    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (prev[0]->requires_grad)
    {
        assert(prev[0]->grad && "Previous tensor 0 gradient cannot be NULL if requires_grad");
        assert(prev[0]->grad->data &&
               "Previous tensor 0 gradient data cannot be NULL if requires_grad");
        assert(prev[0]->grad->data->data &&
               "Previous tensor 0 gradient data pointer cannot be NULL if requires_grad");
        sub_grad_kernel<<<num_blocks, num_threads_per_block>>>(out->grad->data->data,
                                                               prev[0]->grad->data->data, N);
        CHECK_CUDA();
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
    else
    {
        assert(prev[0] && "Previous tensor 0 cannot be NULL");
        assert(prev[0]->data && "Previous tensor 0 data cannot be NULL");
        assert(prev[0]->data->data && "Previous tensor 0 data pointer cannot be NULL");
        assert(prev[1] && "Previous tensor 1 cannot be NULL");
        assert(prev[1]->data && "Previous tensor 1 data cannot be NULL");
        assert(prev[1]->data->data && "Previous tensor 1 data pointer cannot be NULL");

        if (prev[0]->requires_grad)
        {
            assert(prev[0]->grad && "Previous tensor 0 gradient cannot be NULL if requires_grad");
            assert(prev[0]->grad->data &&
                   "Previous tensor 0 gradient data cannot be NULL if requires_grad");
            assert(prev[0]->grad->data->data &&
                   "Previous tensor 0 gradient data pointer cannot be NULL if requires_grad");
            mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, prev[0]->grad->data->data, prev[1]->data->data, N);
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad)
        {
            assert(prev[1]->grad && "Previous tensor 1 gradient cannot be NULL if requires_grad");
            assert(prev[1]->grad->data &&
                   "Previous tensor 1 gradient data cannot be NULL if requires_grad");
            assert(prev[1]->grad->data->data &&
                   "Previous tensor 1 gradient data pointer cannot be NULL if requires_grad");
            mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, prev[1]->grad->data->data, prev[0]->data->data, N);
            CHECK_CUDA();
        }
    }
}
void pow_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("pow_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert((n_prev == 1 || n_prev == 2) && "n_prev must be 1 or 2 for pow_grad_op_cuda");

    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1) // prev[0] ** scalar
    {
        assert(extras && "Extras (scalar value) cannot be NULL for scalar power");
        float scalar_power = *((float*) extras);
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
            scalar_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, prev[0]->data->data, prev[0]->grad->data->data, scalar_power,
                N);
            CHECK_CUDA();
        }
    }
    else // base ** power
    {
        assert(prev[0] && "Previous tensor 0 (base) cannot be NULL");
        assert(prev[0]->data && "Previous tensor 0 (base) data cannot be NULL");
        assert(prev[0]->data->data && "Previous tensor 0 (base) data pointer cannot be NULL");
        assert(prev[1] && "Previous tensor 1 (power) cannot be NULL");
        assert(prev[1]->data && "Previous tensor 1 (power) data cannot be NULL");
        assert(prev[1]->data->data && "Previous tensor 1 (power) data pointer cannot be NULL");

        if (prev[0]->requires_grad) // gradient for base
        {
            assert(prev[0]->grad && "Previous tensor 0 gradient cannot be NULL if requires_grad");
            assert(prev[0]->grad->data &&
                   "Previous tensor 0 gradient data cannot be NULL if requires_grad");
            assert(prev[0]->grad->data->data &&
                   "Previous tensor 0 gradient data pointer cannot be NULL if requires_grad");
            base_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, prev[0]->data->data, prev[0]->grad->data->data,
                prev[1]->data->data, prev[1]->data->data, N);
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad) // gradient for power
        {
            assert(prev[1]->grad && "Previous tensor 1 gradient cannot be NULL if requires_grad");
            assert(prev[1]->grad->data &&
                   "Previous tensor 1 gradient data cannot be NULL if requires_grad");
            assert(prev[1]->grad->data->data &&
                   "Previous tensor 1 gradient data pointer cannot be NULL if requires_grad");
            exponent_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, out->data->data, prev[0]->data->data,
                prev[1]->grad->data->data, N);
            CHECK_CUDA();
        }
    }
}

void div_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("div_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert((n_prev == 1 || n_prev == 2) && "n_prev must be 1 or 2 for div_grad_op_cuda");

    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1) // prev[0] / scalar
    {
        assert(extras && "Extras (scalar value) cannot be NULL for scalar division");
        float scalar_denominator = *((float*) extras);
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
            scalar_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, prev[0]->grad->data->data, scalar_denominator, N);
            CHECK_CUDA();
        }
    }
    else
    {
        assert(prev[0] && "Previous tensor 0 (numerator) cannot be NULL");
        assert(prev[0]->data && "Previous tensor 0 (numerator) data cannot be NULL");
        assert(prev[0]->data->data && "Previous tensor 0 (numerator) data pointer cannot be NULL");
        assert(prev[1] && "Previous tensor 1 (denominator) cannot be NULL");
        assert(prev[1]->data && "Previous tensor 1 (denominator) data cannot be NULL");
        assert(prev[1]->data->data &&
               "Previous tensor 1 (denominator) data pointer cannot be NULL");

        if (prev[0]->requires_grad)
        {
            assert(prev[0]->grad && "Previous tensor 0 gradient cannot be NULL if requires_grad");
            assert(prev[0]->grad->data &&
                   "Previous tensor 0 gradient data cannot be NULL if requires_grad");
            assert(prev[0]->grad->data->data &&
                   "Previous tensor 0 gradient data pointer cannot be NULL if requires_grad");
            numerator_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, prev[0]->grad->data->data, prev[1]->data->data, N);
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad)
        {
            assert(prev[1]->grad && "Previous tensor 1 gradient cannot be NULL if requires_grad");
            assert(prev[1]->grad->data &&
                   "Previous tensor 1 gradient data cannot be NULL if requires_grad");
            assert(prev[1]->grad->data->data &&
                   "Previous tensor 1 gradient data pointer cannot be NULL if requires_grad");
            denominator_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, out->data->data, prev[1]->grad->data->data,
                prev[1]->data->data, N);
            CHECK_CUDA();
        }
    }
}
void rdiv_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("rdiv_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert((n_prev == 1 || n_prev == 2) && "n_prev must be 1 or 2 for rdiv_grad_op_cuda");

    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1)
    {
        assert(extras && "Extras (scalar value) cannot be NULL for scalar rdiv");
        float scalar_numerator = *((float*) extras);
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
            scalar_rdiv_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, out->data->data, prev[0]->grad->data->data, scalar_numerator,
                prev[0]->data->data, N);
            CHECK_CUDA();
        }
    }
    else
    {
        assert(prev[0] && "Previous tensor 0 cannot be NULL");
        assert(prev[0]->data && "Previous tensor 0 data cannot be NULL");
        assert(prev[0]->data->data && "Previous tensor 0 data pointer cannot be NULL");
        assert(prev[1] && "Previous tensor 1 cannot be NULL");
        assert(prev[1]->data && "Previous tensor 1 data cannot be NULL");
        assert(prev[1]->data->data && "Previous tensor 1 data pointer cannot be NULL");

        if (prev[0]->requires_grad)
        {
            assert(prev[0]->grad && "Previous tensor 0 gradient cannot be NULL if requires_grad");
            assert(prev[0]->grad->data &&
                   "Previous tensor 0 gradient data cannot be NULL if requires_grad");
            assert(prev[0]->grad->data->data &&
                   "Previous tensor 0 gradient data pointer cannot be NULL if requires_grad");
            denominator_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, out->data->data, prev[0]->grad->data->data,
                prev[0]->data->data, N);
            CHECK_CUDA();
        }

        if (prev[1]->requires_grad)
        {
            assert(prev[1]->grad && "Previous tensor 1 gradient cannot be NULL if requires_grad");
            assert(prev[1]->grad->data &&
                   "Previous tensor 1 gradient data cannot be NULL if requires_grad");
            assert(prev[1]->grad->data->data &&
                   "Previous tensor 1 gradient data pointer cannot be NULL if requires_grad");
            numerator_div_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, prev[1]->grad->data->data, prev[0]->data->data, N);
            CHECK_CUDA();
        }
    }
}
void matmul_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("matmul_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 2 && "matmul_grad_op_cuda: Expected 2 previous tensors.");

    MatMulBackwardExtras* matmul_extras = (MatMulBackwardExtras*) extras;
    assert(matmul_extras && "matmul_grad_op_cuda: extras is null.");

    Tensor* A = prev[0];
    Tensor* B = prev[1];

    assert(A && "Tensor A cannot be NULL");
    assert(B && "Tensor B cannot be NULL");
    assert(A->data && "Tensor A data cannot be NULL");
    assert(A->data->data && "Tensor A data pointer cannot be NULL");
    assert(B->data && "Tensor B data cannot be NULL");
    assert(B->data->data && "Tensor B data pointer cannot be NULL");

    assert(A->ndim >= 2 && "Tensor A must have at least 2 dimensions");
    assert(B->ndim >= 2 && "Tensor B must have at least 2 dimensions");
    assert(out->ndim >= 2 && "Output tensor must have at least 2 dimensions");

    assert(A->shape && "Tensor A shape cannot be NULL");
    assert(B->shape && "Tensor B shape cannot be NULL");
    assert(out->shape && "Output tensor shape cannot be NULL");

    assert(A->strides && "Tensor A strides cannot be NULL");
    assert(B->strides && "Tensor B strides cannot be NULL");
    assert(out->strides && "Output tensor strides cannot be NULL");

    // Dimension compatibility check
    assert(A->shape[A->ndim - 1] == B->shape[B->ndim - 2] &&
           "Dimension mismatch for matrix multiplication");

    bool is_A_batched = A->ndim > 2;
    bool is_B_batched = B->ndim > 2;
    bool is_out_batched = out->ndim > 2;

    int N_final = matmul_extras->N;
    int K_final = matmul_extras->K;
    int M_final = matmul_extras->M;

    // Validate output dimensions match expected result
    assert(out->shape[out->ndim - 2] == N_final && "Output row dimension mismatch");
    assert(out->shape[out->ndim - 1] == M_final && "Output column dimension mismatch");

    int B_dim = 1;
    if (is_out_batched)
    {
        for (int i = 0; i < out->ndim - 2; ++i)
        {
            B_dim *= out->shape[i];
        }
    }

    if (A->requires_grad)
    {
        assert(A->grad && "Tensor A gradient cannot be NULL if requires_grad");
        assert(A->grad->data && "Tensor A gradient data cannot be NULL if requires_grad");
        assert(A->grad->data->data &&
               "Tensor A gradient data pointer cannot be NULL if requires_grad");

        dim3 block_matmul(TILE_DIM, TILE_DIM);
        dim3 grid_matmul((K_final + block_matmul.x - 1) / block_matmul.x,
                         (N_final + block_matmul.y - 1) / block_matmul.y, B_dim);

        // Gradient for A: dL/dA = dL/dOut @ B.T
        // matmul_grad_kernel(lhs, rhs, grad, B, N, P, K, transpose_lhs, transpose_rhs)
        // lhs = out->grad, rhs = B->data, grad = A->grad
        // N = N_final, P = K_final, K = M_final
        matmul_grad_kernel<<<grid_matmul, block_matmul>>>(
            out->grad->data->data, B->data->data, A->grad->data->data, B_dim, N_final, K_final,
            M_final, false, true, is_out_batched, is_B_batched, is_A_batched);
        CHECK_CUDA();
    }

    if (B->requires_grad)
    {
        assert(B->grad && "Tensor B gradient cannot be NULL if requires_grad");
        assert(B->grad->data && "Tensor B gradient data cannot be NULL if requires_grad");
        assert(B->grad->data->data &&
               "Tensor B gradient data pointer cannot be NULL if requires_grad");

        dim3 block_matmul(TILE_DIM, TILE_DIM);
        dim3 grid_matmul((M_final + block_matmul.x - 1) / block_matmul.x,
                         (K_final + block_matmul.y - 1) / block_matmul.y, B_dim);

        // Gradient for B: dL/dB = A.T @ dL/dOut
        // lhs = A->data, rhs = out->grad, grad = B->grad
        // N = K_final, P = M_final, K = N_final
        matmul_grad_kernel<<<grid_matmul, block_matmul>>>(
            A->data->data, out->grad->data->data, B->grad->data->data, B_dim, K_final, M_final,
            N_final, true, false, is_A_batched, is_out_batched, is_B_batched);
        CHECK_CUDA();
    }
}
void conv2d_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("conv2d_grad_op_cuda: Entering function with n_prev=%d", n_prev);
    assert(out && "Output tensor cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 2 && "conv2d_grad_op_cuda: Expected 2 previous tensors.");
    assert(prev[0] && "Input tensor cannot be NULL");
    assert(prev[1] && "Kernel tensor cannot be NULL");
    assert(extras && "Extras cannot be NULL");
    LOG_WARN("conv2d_grad_op_cuda: CUDA implementation not available yet.");
}
void dot_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("dot_grad_op_cuda: Entering function with n_prev=%d", n_prev);
    assert(out && "Output tensor cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert(n_prev == 2 && "dot_grad_op_cuda: Expected 2 previous tensors.");
    assert(prev[0] && "Input tensor A cannot be NULL");
    assert(prev[1] && "Input tensor B cannot be NULL");
    LOG_WARN("dot_grad_op_cuda: CUDA implementation not available yet.");
}
