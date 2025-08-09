#include "autograd/ops.h"
#include "tensor.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void conv2d_backward_kernel_grad_k(const float* grad_out_data, const float* a_data, float* grad_k_data,
                                              int N, int C_in, int H_in, int W_in,
                                              int C_out, int KH, int KW,
                                              int stride, int padding,
                                              int H_out, int W_out,
                                              const int64_t* a_strides, const int64_t* k_strides, const int64_t* grad_out_strides) {
    int kw = blockIdx.x * blockDim.x + threadIdx.x;
    int kh = blockIdx.y * blockDim.y + threadIdx.y;
    int c_in = blockIdx.z % C_in;
    int c_out = blockIdx.z / C_in;

    if (kw >= KW || kh >= KH || c_in >= C_in || c_out >= C_out) return;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                int h_in = h_out * stride + kh - padding;
                int w_in = w_out * stride + kw - padding;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int64_t a_idx = n * a_strides[0] + c_in * a_strides[1] + h_in * a_strides[2] + w_in * a_strides[3];
                    int64_t grad_out_idx = n * grad_out_strides[0] + c_out * grad_out_strides[1] + h_out * grad_out_strides[2] + w_out * grad_out_strides[3];
                    acc += a_data[a_idx] * grad_out_data[grad_out_idx];
                }
            }
        }
    }
    int64_t grad_k_idx = c_out * k_strides[0] + c_in * k_strides[1] + kh * k_strides[2] + kw * k_strides[3];
    atomicAdd(&grad_k_data[grad_k_idx], acc);
}

__global__ void conv2d_backward_kernel_grad_a(const float* grad_out_data, const float* k_data, float* grad_a_data,
                                              int N, int C_in, int H_in, int W_in,
                                              int C_out, int KH, int KW,
                                              int stride, int padding,
                                              int H_out, int W_out,
                                              const int64_t* a_strides, const int64_t* k_strides, const int64_t* grad_out_strides) {
    int w_in = blockIdx.x * blockDim.x + threadIdx.x;
    int h_in = blockIdx.y * blockDim.y + threadIdx.y;
    int c_in = blockIdx.z % C_in;
    int n = blockIdx.z / C_in;
    
    if (w_in >= W_in || h_in >= H_in || c_in >= C_in || n >= N) return;

    float acc = 0.0f;
    // This is a "full" convolution
    for (int c_out = 0; c_out < C_out; ++c_out) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                int kh = h_in + padding - h_out * stride;
                int kw = w_in + padding - w_out * stride;

                if (kh >= 0 && kh < KH && kw >= 0 && kw < KW) {
                    int64_t grad_out_idx = n * grad_out_strides[0] + c_out * grad_out_strides[1] + h_out * grad_out_strides[2] + w_out * grad_out_strides[3];
                    int64_t k_idx = c_out * k_strides[0] + c_in * k_strides[1] + kh * k_strides[2] + kw * k_strides[3];
                    acc += grad_out_data[grad_out_idx] * k_data[k_idx];
                }
            }
        }
    }
    int64_t grad_a_idx = n * a_strides[0] + c_in * a_strides[1] + h_in * a_strides[2] + w_in * a_strides[3];
    atomicAdd(&grad_a_data[grad_a_idx], acc);
}

void CudaAutograd::conv2d(const Tensor& out, std::vector<Tensor>& prev, int stride, int padding) {
    Tensor t = out;
    Tensor& a = prev[0];
    Tensor& kernel = prev[1];

    const float* grad_out_data = static_cast<const float*>(t.grad_ptr().get());
    const float* a_data = static_cast<const float*>(a.data_ptr().get());
    const float* k_data = static_cast<const float*>(kernel.data_ptr().get());

    const auto& a_shape = a.shape();
    const auto& k_shape = kernel.shape();
    const auto& out_shape = out.shape();

    const int N = a_shape[0], C_in = a_shape[1], H_in = a_shape[2], W_in = a_shape[3];
    const int C_out = k_shape[0], KH = k_shape[2], KW = k_shape[3];
    const int H_out = out_shape[2], W_out = out_shape[3];

    if (kernel.requires_grad()) {
        float* grad_k_data = static_cast<float*>(kernel.grad_ptr().get());
        
        auto d_a_strides = copy_strides_to_device(a.strides());
        auto d_k_strides = copy_strides_to_device(kernel.strides());
        auto d_grad_out_strides = copy_strides_to_device(out.strides());

        dim3 threads(8, 8);
        dim3 blocks(
            (KW + threads.x - 1) / threads.x,
            (KH + threads.y - 1) / threads.y,
            C_out * C_in
        );

        conv2d_backward_kernel_grad_k<<<blocks, threads>>>(
            grad_out_data, a_data, grad_k_data,
            N, C_in, H_in, W_in, C_out, KH, KW,
            stride, padding, H_out, W_out,
            d_a_strides.get(), d_k_strides.get(), d_grad_out_strides.get()
        );
        CUDA_CHECK(cudaGetLastError());
    }

    if (a.requires_grad()) {
        float* grad_a_data = static_cast<float*>(a.grad_ptr().get());

        auto d_a_strides = copy_strides_to_device(a.strides());
        auto d_k_strides = copy_strides_to_device(kernel.strides());
        auto d_grad_out_strides = copy_strides_to_device(out.strides());
        
        dim3 threads(16, 16);
        dim3 blocks(
            (W_in + threads.x - 1) / threads.x,
            (H_in + threads.y - 1) / threads.y,
            N * C_in
        );

        conv2d_backward_kernel_grad_a<<<blocks, threads>>>(
            grad_out_data, k_data, grad_a_data,
            N, C_in, H_in, W_in, C_out, KH, KW,
            stride, padding, H_out, W_out,
            d_a_strides.get(), d_k_strides.get(), d_grad_out_strides.get()
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

