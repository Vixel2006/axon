#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "init.h"
#include <vector>
#include <complex>
#include <numeric>
#include <cmath>
#include <omp.h>

Tensor CpuOps::conv2d(const Tensor& a, const Tensor& kernel, int stride, int padding) {
    const std::vector<int64_t>& in_shape = a.shape();
    const std::vector<int64_t>& kernel_shape = kernel.shape();

    const int64_t C_in = in_shape.size() > 2 ? in_shape[in_shape.size() - 3] : 1;
    const int64_t H_in = in_shape[in_shape.size() - 2];
    const int64_t W_in = in_shape[in_shape.size() - 1];
    
    const int64_t C_out = kernel_shape[0];
    const int64_t H_k = kernel_shape[kernel_shape.size() - 2];
    const int64_t W_k = kernel_shape[kernel_shape.size() - 1];

    const int64_t H_out = (H_in + 2 * padding - H_k) / stride + 1;
    const int64_t W_out = (W_in + 2 * padding - W_k) / stride + 1;

    const int64_t H_fft = next_power_of_2(H_in + H_k - 1);
    const int64_t W_fft = next_power_of_2(W_in + W_k - 1);
    const int64_t fft_slice_size = H_fft * W_fft;

    int64_t batch_size = 1;
    if (in_shape.size() > 3) {
        batch_size = in_shape[0];
    }

    std::vector<int64_t> out_shape = {batch_size, C_out, H_out, W_out};
    Tensor out = zeros(out_shape, deviceToString(a.device()), a.requires_grad());

    float* a_data = static_cast<float*>(a.data_ptr().get());
    float* kernel_data = static_cast<float*>(kernel.data_ptr().get());
    float* out_data = static_cast<float*>(out.data_ptr().get());
    
    const int64_t in_channel_size = H_in * W_in;
    const int64_t kernel_channel_size = H_k * W_k;
    const int64_t out_channel_size = H_out * W_out;

    std::vector<std::vector<std::complex<double>>> all_kernels_fft(C_out * C_in);
    for (int64_t c_out = 0; c_out < C_out; ++c_out) {
        for (int64_t c_in = 0; c_in < C_in; ++c_in) {
            int64_t kernel_idx = c_out * C_in + c_in;
            float* kernel_slice_ptr = kernel_data + kernel_idx * kernel_channel_size;
            
            all_kernels_fft[kernel_idx].assign(fft_slice_size, {0.0, 0.0});
            
            for (int64_t h = 0; h < H_k; ++h) {
                for (int64_t w = 0; w < W_k; ++w) {
                    all_kernels_fft[kernel_idx][h * W_fft + w] = std::complex<double>(kernel_slice_ptr[h * W_k + w], 0.0);
                }
            }
            fft2d(all_kernels_fft[kernel_idx], H_fft, W_fft);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t c_out = 0; c_out < C_out; ++c_out) {
            std::vector<std::complex<double>> full_conv_result(fft_slice_size, {0.0, 0.0});

            for (int64_t c_in = 0; c_in < C_in; ++c_in) {
                std::vector<std::complex<double>> in_fft(fft_slice_size, {0.0, 0.0});
                
                float* in_slice_ptr = a_data + (b * C_in + c_in) * in_channel_size;

                for (int64_t h = 0; h < H_in; ++h) {
                    for (int64_t w = 0; w < W_in; ++w) {
                        in_fft[h * W_fft + w] = std::complex<double>(in_slice_ptr[h * W_in + w], 0.0);
                    }
                }
                
                fft2d(in_fft, H_fft, W_fft);
                
                const int64_t kernel_idx = c_out * C_in + c_in;
                for (int i = 0; i < fft_slice_size; ++i) {
                    in_fft[i] *= all_kernels_fft[kernel_idx][i];
                }
                
                ifft2d(in_fft, H_fft, W_fft);

                for (int i = 0; i < fft_slice_size; ++i) {
                    full_conv_result[i] += in_fft[i];
                }
            }

            float* out_slice_ptr = out_data + (b * C_out + c_out) * out_channel_size;
            for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                    int64_t h_src = (h_out * stride - padding) + (H_k - 1);
                    int64_t w_src = (w_out * stride - padding) + (W_k - 1);
                    
                    if (h_src >= 0 && h_src < H_fft && w_src >= 0 && w_src < W_fft) {
                        out_slice_ptr[h_out * W_out + w_out] = full_conv_result[h_src * W_fft + w_src].real();
                    }
                }
            }
        }
    }

    return out;
}


