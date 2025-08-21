#include "engine/ops.h"
#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include "autograd/ops.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <vector>

Tensor CudaOps::conv2d(const Tensor& a, const Tensor& kernel, int stride, int padding) {
    const std::vector<int64_t>& in_shape = a.shape();
    const std::vector<int64_t>& kernel_shape = kernel.shape();

    const int64_t N = in_shape[0];
    const int64_t C_in = in_shape[1];
    const int64_t H_in = in_shape[2];
    const int64_t W_in = in_shape[3];

    const int64_t C_out = kernel_shape[0];
    const int64_t H_k = kernel_shape[2];
    const int64_t W_k = kernel_shape[3];

    const int64_t H_out = (H_in + 2 * padding - H_k) / stride + 1;
    const int64_t W_out = (W_in + 2 * padding - W_k) / stride + 1;

    Tensor out({N, C_out, H_out, W_out}, a.dtype(), deviceToString(a.device()), a.requires_grad());

    const float* d_a = static_cast<const float*>(a.data_ptr().get());
    const float* d_kernel = static_cast<const float*>(kernel.data_ptr().get());
    float* d_out = static_cast<float*>(out.data_ptr().get());

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    const int64_t K = C_in * H_k * W_k;
    const int64_t L = H_out * W_out;
    const int64_t M = C_out;

    float* d_col_buffer;
    CUDA_CHECK(cudaMalloc(&d_col_buffer, K * L * sizeof(float)));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int n = 0; n < N; ++n) {
        const float* d_a_n = d_a + n * (C_in * H_in * W_in);
        float* d_out_n = d_out + n * (C_out * H_out * W_out);

        int threads_per_block = 256;
        int num_blocks = (L + threads_per_block - 1) / threads_per_block;
        /*
        im2col_kernel<<<num_blocks, threads_per_block>>>(
            d_a_n, d_col_buffer, C_in, H_in, W_in, H_k, W_k, H_out, W_out, stride, padding
        );
        CUDA_CHECK(cudaGetLastError());
        */
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 L, M, K,
                                 &alpha,
                                 d_col_buffer, L,
                                 d_kernel, K,
                                 &beta,
                                 d_out_n, L));
    }

    CUDA_CHECK(cudaFree(d_col_buffer));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    if (out.requires_grad()) {
      auto backward_fn = [stride, padding](Tensor& out, std::vector<Tensor>& prev) {
        CudaAutograd::conv2d(out, prev, stride, padding);
      };
      out.set_ctx({a, kernel}, backward_fn);
    }

    return out;
}
