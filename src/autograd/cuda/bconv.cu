#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <vector>


void CudaAutograd::conv2d(const Tensor& out, std::vector<Tensor>& prev) {
    if (prev.size() != 2) {
        throw std::runtime_error("conv2d backward expects 2 previous tensors (input and kernel)");
    }

    Tensor t = out;
    Tensor& a = prev[0];
    Tensor& kernel = prev[1];

    if (!a.requires_grad() && !kernel.requires_grad()) {
        return;
    }

    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    if (!out_grad_p) {
        throw std::runtime_error("Output gradient tensor is null for conv2d backward (CUDA).");
    }

    const auto& a_shape = a.shape();
    const auto& kernel_shape = kernel.shape();
    const auto& out_shape = out.shape();

    const int H_in = a_shape[a_shape.size() - 2];
    const int W_in = a_shape[a_shape.size() - 1];
    const int H_k = kernel_shape[kernel_shape.size() - 2];
    const int W_k = kernel_shape[kernel_shape.size() - 1];
    const int H_out = out_shape[out_shape.size() - 2];
    const int W_out = out_shape[out_shape.size() - 1];

    int stride = 1;
    int padding = 0;

    const int H_fft = next_power_of_2(H_in + H_k - 1);
    const int W_fft = next_power_of_2(W_in + W_k - 1);
    const int W_fft_complex = (W_fft / 2) + 1;
    const size_t complex_elements = W_fft_complex * H_fft;

    float *d_padded_grad, *d_padded_a, *d_padded_kernel_rot;
    cufftComplex *d_grad_fft, *d_a_fft, *d_kernel_fft;

    const size_t real_size = W_fft * H_fft * sizeof(float);
    const size_t complex_size = complex_elements * sizeof(cufftComplex);

    CUDA_CHECK(cudaMalloc(&d_padded_grad, real_size));
    CUDA_CHECK(cudaMalloc(&d_grad_fft, complex_size));

    cufftHandle plan_r2c, plan_c2r;
    checkCufftErrors(cufftPlan2d(&plan_r2c, H_fft, W_fft, CUFFT_R2C));
    checkCufftErrors(cufftPlan2d(&plan_c2r, H_fft, W_fft, CUFFT_C2R));

    CUDA_CHECK(cudaMemset(d_padded_grad, 0, real_size));
    dim3 grad_pad_block(16, 16);
    dim3 grad_pad_grid((W_out + grad_pad_block.x - 1) / grad_pad_block.x, 
                       (H_out + grad_pad_block.y - 1) / grad_pad_block.y);
    pad_grad_kernel<<<grad_pad_grid, grad_pad_block>>>(
        out_grad_p, d_padded_grad, W_out, H_out, W_fft, H_fft, W_k, H_k, stride, padding
    );
    checkCufftErrors(cufftExecR2C(plan_r2c, (cufftReal*)d_padded_grad, d_grad_fft));

    if (a.requires_grad()) {
        float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
        const float* kernel_p = static_cast<const float*>(kernel.data_ptr().get());

        CUDA_CHECK(cudaMalloc(&d_padded_kernel_rot, real_size));
        CUDA_CHECK(cudaMalloc(&d_kernel_fft, complex_size));

        CUDA_CHECK(cudaMemset(d_padded_kernel_rot, 0, real_size));
        dim3 k_pad_block(16, 16);
        dim3 k_pad_grid((W_k + k_pad_block.x - 1) / k_pad_block.x, (H_k + k_pad_block.y - 1) / k_pad_block.y);
        pad_and_rotate_kernel<<<k_pad_grid, k_pad_block>>>(kernel_p, d_padded_kernel_rot, W_k, H_k, W_fft, H_fft);
        checkCufftErrors(cufftExecR2C(plan_r2c, (cufftReal*)d_padded_kernel_rot, d_kernel_fft));

        const float scale = 1.0f / (W_fft * H_fft);
        int threads_per_block_mult = 256;
        int blocks_mult = (complex_elements + threads_per_block_mult - 1) / threads_per_block_mult;
        complex_mult_and_scale_kernel<<<blocks_mult, threads_per_block_mult>>>(d_kernel_fft, d_grad_fft, complex_elements, scale);
        
        checkCufftErrors(cufftExecC2R(plan_c2r, d_kernel_fft, (cufftReal*)d_padded_kernel_rot));
        
        dim3 crop_block(16, 16);
        dim3 crop_grid((W_in + crop_block.x - 1) / crop_block.x, (H_in + crop_block.y - 1) / crop_block.y);
        crop_and_add_kernel<<<crop_grid, crop_block>>>(d_padded_kernel_rot, a_grad_p, W_in, H_in, W_fft);

        CUDA_CHECK(cudaFree(d_padded_kernel_rot));
        CUDA_CHECK(cudaFree(d_kernel_fft));
    }

    if (kernel.requires_grad()) {
        float* kernel_grad_p = static_cast<float*>(kernel.grad_ptr().get());
        const float* a_p = static_cast<const float*>(a.data_ptr().get());

        CUDA_CHECK(cudaMalloc(&d_padded_a, real_size));
        CUDA_CHECK(cudaMalloc(&d_a_fft, complex_size));

        CUDA_CHECK(cudaMemset(d_padded_a, 0, real_size));
        dim3 a_pad_block(16, 16);
        dim3 a_pad_grid((W_fft + a_pad_block.x - 1) / a_pad_block.x, (H_fft + a_pad_block.y - 1) / a_pad_block.y);
        pad_kernel<<<a_pad_grid, a_pad_block>>>(a_p, d_padded_a, W_in, H_in, W_fft, H_fft); // Re-use forward pass kernel
        checkCufftErrors(cufftExecR2C(plan_r2c, (cufftReal*)d_padded_a, d_a_fft));

        const float scale = 1.0f / (W_fft * H_fft);
        int threads_per_block_mult = 256;
        int blocks_mult = (complex_elements + threads_per_block_mult - 1) / threads_per_block_mult;
        complex_mult_and_scale_kernel<<<blocks_mult, threads_per_block_mult>>>(d_a_fft, d_grad_fft, complex_elements, scale);
        
        checkCufftErrors(cufftExecC2R(plan_c2r, d_a_fft, (cufftReal*)d_padded_a));
        
        dim3 crop_block(16, 16);
        dim3 crop_grid((W_k + crop_block.x - 1) / crop_block.x, (H_k + crop_block.y - 1) / crop_block.y);
        crop_and_add_kernel<<<crop_grid, crop_block>>>(d_padded_a, kernel_grad_p, W_k, H_k, W_fft);
        
        CUDA_CHECK(cudaFree(d_padded_a));
        CUDA_CHECK(cudaFree(d_a_fft));
    }

    checkCufftErrors(cufftDestroy(plan_r2c));
    checkCufftErrors(cufftDestroy(plan_c2r));
    CUDA_CHECK(cudaFree(d_padded_grad));
    CUDA_CHECK(cudaFree(d_grad_fft));
    CUDA_CHECK(cudaGetLastError());
}
