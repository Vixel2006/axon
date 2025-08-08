#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include <vector>

Tensor CpuOps::conv2d(const Tensor& a, const Tensor& kernel, int stride, int padding) {
    const auto& in_shape = a.shape();
    const auto& k_shape = kernel.shape();

    const int64_t N = in_shape[0];
    const int64_t C_in = in_shape[1];
    const int64_t H_in = in_shape[2];
    const int64_t W_in = in_shape[3];

    const int64_t C_out = k_shape[0];
    const int64_t KH = k_shape[2];
    const int64_t KW = k_shape[3];

    const int64_t H_out = (H_in + 2 * padding - KH) / stride + 1;
    const int64_t W_out = (W_in + 2 * padding - KW) / stride + 1;

    std::vector<int64_t> out_shape = {N, C_out, H_out, W_out};
    Tensor result(out_shape, a.dtype(), deviceToString(a.device()), a.requires_grad() || kernel.requires_grad());

    const float* a_data = static_cast<const float*>(a.data_ptr().get());
    const float* k_data = static_cast<const float*>(kernel.data_ptr().get());
    float* res_data = static_cast<float*>(result.data_ptr().get());
    
    const auto& a_strides = a.strides();
    const auto& k_strides = kernel.strides();
    const auto& res_strides = result.strides();

    #pragma omp parallel for collapse(4) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c_out = 0; c_out < C_out; ++c_out) {
            for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                    
                    float acc = 0.0f;
                    for (int64_t c_in = 0; c_in < C_in; ++c_in) {
                        for (int64_t kh = 0; kh < KH; ++kh) {
                            for (int64_t kw = 0; kw < KW; ++kw) {
                                int64_t h_in = h_out * stride + kh - padding;
                                int64_t w_in = w_out * stride + kw - padding;

                                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                    int64_t a_idx = n * a_strides[0] + c_in * a_strides[1] + h_in * a_strides[2] + w_in * a_strides[3];
                                    int64_t k_idx = c_out * k_strides[0] + c_in * k_strides[1] + kh * k_strides[2] + kw * k_strides[3];
                                    acc += a_data[a_idx] * k_data[k_idx];
                                }
                            }
                        }
                    }
                    res_data[n * res_strides[0] + c_out * res_strides[1] + h_out * res_strides[2] + w_out * res_strides[3]] = acc;
                }
            }
        }
    }

    if (result.requires_grad()) {
        result.set_ctx({a, kernel}, [stride, padding](const Tensor& out, std::vector<Tensor>& prev) {
            CpuAutograd::conv2d(out, prev, stride, padding);
        });
    }
    return result;
}
