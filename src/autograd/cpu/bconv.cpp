#include "autograd/ops.h"
#include "tensor.h"
#include <vector>
#include "utils.h"

void CpuAutograd::conv2d(const Tensor& out, std::vector<Tensor>& prev, int stride, int padding) {
    Tensor t = out;
    Tensor& a = prev[0];
    Tensor& kernel = prev[1];

    const float* grad_out_data = static_cast<const float*>(t.grad_ptr().get());
    const float* a_data = static_cast<const float*>(a.data_ptr().get());
    const float* kernel_data = static_cast<const float*>(kernel.data_ptr().get());

    const auto& a_shape = a.shape();
    const auto& k_shape = kernel.shape();
    const auto& out_shape = out.shape();

    const int64_t N = a_shape[0], C_in = a_shape[1], H_in = a_shape[2], W_in = a_shape[3];
    const int64_t C_out = k_shape[0], KH = k_shape[2], KW = k_shape[3];
    const int64_t H_out = out_shape[2], W_out = out_shape[3];
    
    const auto& a_strides = a.strides();
    const auto& k_strides = kernel.strides();
    const auto& out_grad_strides = t.strides();

    if (kernel.requires_grad()) {
        float* grad_k_data = static_cast<float*>(kernel.grad_ptr().get());
        
        #pragma omp parallel for collapse(4) schedule(static)
        for (int64_t c_out = 0; c_out < C_out; ++c_out) {
            for (int64_t c_in = 0; c_in < C_in; ++c_in) {
                for (int64_t kh = 0; kh < KH; ++kh) {
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        float acc = 0.0f;
                        for (int64_t n = 0; n < N; ++n) {
                            for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                                for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                                    int64_t h_in = h_out * stride + kh - padding;
                                    int64_t w_in = w_out * stride + kw - padding;
                                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                        int64_t a_idx = n * a_strides[0] + c_in * a_strides[1] + h_in * a_strides[2] + w_in * a_strides[3];
                                        int64_t grad_out_idx = n * out_grad_strides[0] + c_out * out_grad_strides[1] + h_out * out_grad_strides[2] + w_out * out_grad_strides[3];
                                        acc += a_data[a_idx] * grad_out_data[grad_out_idx];
                                    }
                                }
                            }
                        }
                        int64_t grad_k_idx = c_out * k_strides[0] + c_in * k_strides[1] + kh * k_strides[2] + kw * k_strides[3];
                        grad_k_data[grad_k_idx] += acc;
                    }
                }
            }
        }
    }

    if (a.requires_grad()) {
        float* grad_a_data = static_cast<float*>(a.grad_ptr().get());
        
        #pragma omp parallel for collapse(4) schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c_in = 0; c_in < C_in; ++c_in) {
                for (int64_t h_in = 0; h_in < H_in; ++h_in) {
                    for (int64_t w_in = 0; w_in < W_in; ++w_in) {
                        float acc = 0.0f;
                        for (int64_t c_out = 0; c_out < C_out; ++c_out) {
                            for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                                for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                                    int64_t kh = h_in + padding - h_out * stride;
                                    int64_t kw = w_in + padding - w_out * stride;
                                    
                                    if (kh >= 0 && kh < KH && kw >= 0 && kw < KW) {
                                        int64_t grad_out_idx = n * out_grad_strides[0] + c_out * out_grad_strides[1] + h_out * out_grad_strides[2] + w_out * out_grad_strides[3];
                                        int64_t k_idx = c_out * k_strides[0] + c_in * k_strides[1] + kh * k_strides[2] + kw * k_strides[3];
                                        acc += grad_out_data[grad_out_idx] * kernel_data[k_idx];
                                    }
                                }
                            }
                        }
                        int64_t grad_a_idx = n * a_strides[0] + c_in * a_strides[1] + h_in * a_strides[2] + w_in * a_strides[3];
                        grad_a_data[grad_a_idx] += acc;
                    }
                }
            }
        }
    }
}

