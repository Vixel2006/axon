#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include "autograd/ops.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <vector>

__global__ void complex_mult_accumulate_kernel(cufftComplex* accumulator, const cufftComplex* a, const cufftComplex* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ar = a[idx].x;
        float ai = a[idx].y;
        float br = b[idx].x;
        float bi = b[idx].y;
        
        // C = A * B
        float cr = ar * br - ai * bi;
        float ci = ar * bi + ai * br;

        // Accumulator += C
        atomicAdd(&accumulator[idx].x, cr);
        atomicAdd(&accumulator[idx].y, ci);
    }
}


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

    const int64_t H_fft = next_power_of_2(H_in + H_k - 1);
    const int64_t W_fft = next_power_of_2(W_in + W_k - 1);
    const int64_t W_fft_complex = (W_fft / 2) + 1;

    const size_t fft_real_size = W_fft * H_fft * sizeof(float);
    const size_t fft_complex_size = W_fft_complex * H_fft * sizeof(cufftComplex);
    const int    complex_elements = W_fft_complex * H_fft;

    Tensor out({N, C_out, H_out, W_out}, a.dtype(), deviceToString(a.device()), a.requires_grad());
    float* d_out_full = static_cast<float*>(out.data_ptr().get());

    cufftHandle plan_r2c, plan_c2r;
    checkCufftErrors(cufftPlan2d(&plan_r2c, H_fft, W_fft, CUFFT_R2C));
    checkCufftErrors(cufftPlan2d(&plan_c2r, H_fft, W_fft, CUFFT_C2R));

    float* d_kernel_all = nullptr;
    cufftComplex* d_kernels_fft_all = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kernel_all, kernel.numel()));
    CUDA_CHECK(cudaMalloc(&d_kernels_fft_all, C_out * C_in * fft_complex_size));
    CUDA_CHECK(cudaMemcpy(d_kernel_all, kernel.data_ptr().get(), kernel.numel(), cudaMemcpyHostToDevice));

    float* h_kernel_slice = new float[H_k * W_k];
    float* d_padded_temp;
    CUDA_CHECK(cudaMalloc(&d_padded_temp, fft_real_size));
    
    for (int64_t c_out = 0; c_out < C_out; ++c_out) {
        for (int64_t c_in = 0; c_in < C_in; ++c_in) {
            const int64_t kernel_channel_offset = (c_out * C_in + c_in) * H_k * W_k;
            const int64_t kernel_fft_offset = (c_out * C_in + c_in) * complex_elements;

            pad_kernel<<<dim3((W_fft+15)/16, (H_fft+15)/16), dim3(16,16)>>>(
                d_kernel_all + kernel_channel_offset, d_padded_temp, W_k, H_k, W_fft, H_fft);
            
            checkCufftErrors(cufftExecR2C(plan_r2c, (cufftReal*)d_padded_temp, d_kernels_fft_all + kernel_fft_offset));
        }
    }
    delete[] h_kernel_slice;
    CUDA_CHECK(cudaFree(d_padded_temp));


    float* d_input_all = nullptr;
    float* d_padded_input = nullptr;
    cufftComplex* d_input_fft = nullptr;
    cufftComplex* d_acc_fft = nullptr;
    float* d_conv_result_padded = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input_all, a.numel()));
    CUDA_CHECK(cudaMalloc(&d_padded_input, fft_real_size));
    CUDA_CHECK(cudaMalloc(&d_input_fft, fft_complex_size));
    CUDA_CHECK(cudaMalloc(&d_acc_fft, fft_complex_size));
    CUDA_CHECK(cudaMalloc(&d_conv_result_padded, fft_real_size));
    CUDA_CHECK(cudaMemcpy(d_input_all, a.data_ptr().get(), a.numel(), cudaMemcpyHostToDevice));


    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c_out = 0; c_out < C_out; ++c_out) {
            CUDA_CHECK(cudaMemset(d_acc_fft, 0, fft_complex_size));

            for (int64_t c_in = 0; c_in < C_in; ++c_in) {
                const float* d_current_input_slice = d_input_all + (n * C_in + c_in) * H_in * W_in;
                const cufftComplex* d_current_kernel_fft = d_kernels_fft_all + (c_out * C_in + c_in) * complex_elements;

                pad_kernel<<<dim3((W_fft+15)/16, (H_fft+15)/16), dim3(16,16)>>>(
                    d_current_input_slice, d_padded_input, W_in, H_in, W_fft, H_fft);

                checkCufftErrors(cufftExecR2C(plan_r2c, (cufftReal*)d_padded_input, d_input_fft));

                int threads = 256;
                int blocks = (complex_elements + threads - 1) / threads;
                complex_mult_accumulate_kernel<<<blocks, threads>>>(
                    d_acc_fft, d_input_fft, d_current_kernel_fft, complex_elements);
            }

            checkCufftErrors(cufftExecC2R(plan_c2r, d_acc_fft, (cufftReal*)d_conv_result_padded));
            
            float* d_current_out_slice = d_out_full + (n * C_out + c_out) * H_out * W_out;
            dim3 crop_block_dim(16, 16);
            dim3 crop_grid_dim((W_out + 15) / 16, (H_out + 15) / 16);
            crop_and_stride_kernel<<<crop_grid_dim, crop_block_dim>>>(
                d_conv_result_padded, d_current_out_slice, W_fft, W_out, H_out, W_k, H_k, stride, padding);
        }
    }
    
    CUDA_CHECK(cudaFree(d_kernel_all));
    CUDA_CHECK(cudaFree(d_kernels_fft_all));
    CUDA_CHECK(cudaFree(d_input_all));
    CUDA_CHECK(cudaFree(d_padded_input));
    CUDA_CHECK(cudaFree(d_input_fft));
    CUDA_CHECK(cudaFree(d_acc_fft));
    CUDA_CHECK(cudaFree(d_conv_result_padded));
    checkCufftErrors(cufftDestroy(plan_r2c));
    checkCufftErrors(cufftDestroy(plan_c2r));
    
    return out;
}

