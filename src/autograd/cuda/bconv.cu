#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <vector>

#include <cublas_v2.h>



void CudaAutograd::conv2d(Tensor& out, std::vector<Tensor>& prev, int stride, int padding) {
    Tensor t = out;
    Tensor& a = prev[0];
    Tensor& kernel = prev[1];

    if (!a.requires_grad() && !kernel.requires_grad()) return;

    const std::vector<int64_t>& a_shape = a.shape();
    const std::vector<int64_t>& kernel_shape = kernel.shape();
    const std::vector<int64_t>& out_grad_shape = out.shape();

    const int64_t N = a_shape[0];
    const int64_t C_in = a_shape[1];
    const int64_t H_in = a_shape[2];
    const int64_t W_in = a_shape[3];

    const int64_t C_out = kernel_shape[0];
    const int64_t H_k = kernel_shape[2];
    const int64_t W_k = kernel_shape[3];

    const int64_t H_out = out_grad_shape[2];
    const int64_t W_out = out_grad_shape[3];

    const float* d_a_data = static_cast<const float*>(a.data_ptr().get());
    const float* d_kernel_data = static_cast<const float*>(kernel.data_ptr().get());
    const float* d_out_grad = static_cast<const float*>(t.grad_ptr().get());

    float* d_a_grad = a.requires_grad() ? static_cast<float*>(a.grad_ptr().get()) : nullptr;
    float* d_kernel_grad = kernel.requires_grad() ? static_cast<float*>(kernel.grad_ptr().get()) : nullptr;

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    const int64_t M_k = C_out;
    const int64_t K_k = C_in * H_k * W_k;
    const int64_t L_k = H_out * W_out;

    const float alpha = 1.0f;

    if (kernel.requires_grad()) {
        float* d_col_buffer;
        CUDA_CHECK(cudaMalloc(&d_col_buffer, K_k * L_k * sizeof(float)));

        for (int n = 0; n < N; ++n) {
            const float* d_a_n = d_a_data + n * (C_in * H_in * W_in);
            const float* d_out_grad_n = d_out_grad + n * (C_out * H_out * W_out);
            
            int threads = 512;
            int blocks = (K_k * L_k + threads - 1) / threads;
            im2col_kernel<<<blocks, threads>>>(d_a_n, d_col_buffer, C_in, H_in, W_in, H_k, W_k, H_out, W_out, stride, padding);
            CUDA_CHECK(cudaGetLastError());
            
            float beta = (n == 0) ? 0.0f : 1.0f;
            CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     K_k, M_k, L_k,
                                     &alpha,
                                     d_col_buffer, L_k,
                                     d_out_grad_n, L_k,
                                     &beta,
                                     d_kernel_grad, K_k));
        }
        CUDA_CHECK(cudaFree(d_col_buffer));
    }

    if (a.requires_grad()) {
        float* d_a_grad_col;
        CUDA_CHECK(cudaMalloc(&d_a_grad_col, K_k * L_k * sizeof(float)));

        const float beta = 0.0f;

        for (int n = 0; n < N; ++n) {
            const float* d_out_grad_n = d_out_grad + n * (C_out * H_out * W_out);
            float* d_a_grad_n = d_a_grad + n * (C_in * H_in * W_in);

            CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                     L_k, K_k, M_k,
                                     &alpha,
                                     d_out_grad_n, L_k,
                                     d_kernel_data, K_k,
                                     &beta,
                                     d_a_grad_col, L_k));
            
            CUDA_CHECK(cudaMemset(d_a_grad_n, 0, C_in * H_in * W_in * sizeof(float)));
            int threads = 512;
            int blocks = (K_k * L_k + threads - 1) / threads;
            col2im_kernel<<<blocks, threads>>>(d_a_grad_col, d_a_grad_n, C_in, H_in, W_in, H_k, W_k, H_out, W_out, stride, padding);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaFree(d_a_grad_col));
    }
    
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}
