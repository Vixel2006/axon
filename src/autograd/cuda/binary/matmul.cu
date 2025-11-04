#include "autograd/cuda/binary/common.cuh"
#include "utils/indexing.cuh"
#include "tensor.h"

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

    float* out_grad_data_ptr = (float*) out->grad->data->data;
    float* out_grad_temp_data = NULL;

    if (!is_contiguous(out->grad))
    {
        int num_elements_out_grad = numel(out->grad->shape, out->grad->ndim);
        cudaMalloc((void**) &out_grad_temp_data, num_elements_out_grad * sizeof(float));
        int num_threads_per_block = 256; // Assuming 256 threads per block as in matmul.cu
        int num_blocks = (num_elements_out_grad + num_threads_per_block - 1) / num_threads_per_block;
        copy_non_contiguous_to_contiguous_kernel<<<num_blocks, num_threads_per_block>>>(
            (float*) out->grad->data->data, out_grad_temp_data, out->grad->shape, out->grad->strides,
            out->grad->ndim, num_elements_out_grad);
        out_grad_data_ptr = out_grad_temp_data;
        CHECK_CUDA();
    }

    float* A_data_ptr = (float*) A->data->data;
    float* A_temp_data = NULL;
    if (!is_contiguous(A))
    {
        int num_elements_A = numel(A->shape, A->ndim);
        cudaMalloc((void**) &A_temp_data, num_elements_A * sizeof(float));
        int num_threads_per_block = 256;
        int num_blocks = (num_elements_A + num_threads_per_block - 1) / num_threads_per_block;
        copy_non_contiguous_to_contiguous_kernel<<<num_blocks, num_threads_per_block>>>(
            (float*) A->data->data, A_temp_data, A->shape, A->strides, A->ndim, num_elements_A);
        A_data_ptr = A_temp_data;
        CHECK_CUDA();
    }

    float* B_data_ptr = (float*) B->data->data;
    float* B_temp_data = NULL;
    if (!is_contiguous(B))
    {
        int num_elements_B = numel(B->shape, B->ndim);
        cudaMalloc((void**) &B_temp_data, num_elements_B * sizeof(float));
        int num_threads_per_block = 256;
        int num_blocks = (num_elements_B + num_threads_per_block - 1) / num_threads_per_block;
        copy_non_contiguous_to_contiguous_kernel<<<num_blocks, num_threads_per_block>>>(
            (float*) B->data->data, B_temp_data, B->shape, B->strides, B->ndim, num_elements_B);
        B_data_ptr = B_temp_data;
        CHECK_CUDA();
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

        // Initialize A's gradient to zero before accumulation
        int num_elements_A_grad = numel(A->grad->shape, A->grad->ndim);
        cudaMemset(A->grad->data->data, 0, num_elements_A_grad * sizeof(float));
        CHECK_CUDA();

        // Gradient for A: dL/dA = dL/dOut @ B.T
        // matmul_grad_kernel(lhs, rhs, grad, B, N, P, K, transpose_lhs, transpose_rhs)
        // lhs = out->grad, rhs = B->data, grad = A->grad
        // N = N_final, P = K_final, K = M_final
        matmul_grad_kernel<<<grid_matmul, block_matmul>>>(
            out_grad_data_ptr, B_data_ptr, A->grad->data->data, B_dim, N_final, K_final,
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

        // Initialize B's gradient to zero before accumulation
        int num_elements_B_grad = numel(B->grad->shape, B->grad->ndim);
        cudaMemset(B->grad->data->data, 0, num_elements_B_grad * sizeof(float));
        CHECK_CUDA();

        // Gradient for B: dL/dB = A.T @ dL/dOut
        // lhs = A->data, rhs = out->grad, grad = B->grad
        // N = K_final, P = M_final, K = N_final
        matmul_grad_kernel<<<grid_matmul, block_matmul>>>(
            A_data_ptr, out_grad_data_ptr, B->grad->data->data, B_dim, K_final, M_final,
            N_final, true, false, is_A_batched, is_out_batched, is_B_batched);
        CHECK_CUDA();
    }

    if (out_grad_temp_data)
    {
        cudaFree(out_grad_temp_data);
    }
    if (A_temp_data)
    {
        cudaFree(A_temp_data);
    }
    if (B_temp_data)
    {
        cudaFree(B_temp_data);
    }
}
