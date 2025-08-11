#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include <vector>
#include <complex>
#include <stdexcept>
#include <omp.h>


void CpuAutograd::conv2d(const Tensor& out, std::vector<Tensor>& prev) {
    if (prev.size() != 2) {
        throw std::runtime_error("conv2d_backward expects 2 previous tensors (input and kernel)");
    }

    Tensor& a = prev[0];
    Tensor& kernel = prev[1];
    Tensor t = out;

    const std::vector<int64_t>& in_shape = a.shape();
    const std::vector<int64_t>& kernel_shape = kernel.shape();
    const std::vector<int64_t>& out_grad_shape = t.shape();

    const int64_t H_in = in_shape[in_shape.size() - 2];
    const int64_t W_in = in_shape[in_shape.size() - 1];
    const int64_t H_k = kernel_shape[kernel_shape.size() - 2];
    const int64_t W_k = kernel_shape[kernel_shape.size() - 1];
    const int64_t H_out = out_grad_shape[out_grad_shape.size() - 2];
    const int64_t W_out = out_grad_shape[out_grad_shape.size() - 1];

    const int64_t H_fft = H_in + H_k - 1;
    const int64_t W_fft = W_in + W_k - 1;

    int64_t batch_dims = 1;
    for (size_t i = 0; i < in_shape.size() - 2; ++i) {
        batch_dims *= in_shape[i];
    }

    const float* a_data = static_cast<const float*>(a.data_ptr().get());
    const float* kernel_data = static_cast<const float*>(kernel.data_ptr().get());
    const float* out_grad_data = static_cast<const float*>(t.grad_ptr().get());

    float* a_grad_data = static_cast<float*>(a.grad_ptr().get());
    float* kernel_grad_data = static_cast<float*>(kernel.grad_ptr().get());

    const int64_t in_slice_size = H_in * W_in;
    const int64_t kernel_slice_size = H_k * W_k;
    const int64_t out_slice_size = H_out * W_out;
    const int64_t fft_slice_size = H_fft * W_fft;

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < batch_dims; ++b) {
        std::vector<std::complex<double>> out_grad_fft(fft_slice_size, {0.0, 0.0});
        const float* out_grad_slice_ptr = out_grad_data + b * out_slice_size;
        for (int64_t h = 0; h < H_out; ++h) {
            for (int64_t w = 0; w < W_out; ++w) {
                out_grad_fft[h * W_fft + w] = std::complex<double>(out_grad_slice_ptr[h * W_out + w], 0.0);
            }
        }
        fft2d(out_grad_fft, H_fft, W_fft);

        if (a.requires_grad()) {
            std::vector<std::complex<double>> kernel_fft(fft_slice_size, {0.0, 0.0});
            const float* kernel_slice_ptr = kernel_data + b * kernel_slice_size;
            for (int64_t h = 0; h < H_k; ++h) {
                for (int64_t w = 0; w < W_k; ++w) {
                    kernel_fft[(H_k - 1 - h) * W_fft + (W_k - 1 - w)] = std::complex<double>(kernel_slice_ptr[h * W_k + w], 0.0);
                }
            }
            fft2d(kernel_fft, H_fft, W_fft);

            std::vector<std::complex<double>> a_grad_fft(fft_slice_size);
            for (int i = 0; i < fft_slice_size; ++i) {
                a_grad_fft[i] = out_grad_fft[i] * kernel_fft[i];
            }
            ifft2d(a_grad_fft, H_fft, W_fft);

            float* a_grad_slice_ptr = a_grad_data + b * in_slice_size;
            for (int64_t h = 0; h < H_in; ++h) {
                for (int64_t w = 0; w < W_in; ++w) {
                    a_grad_slice_ptr[h * W_in + w] += a_grad_fft[h * W_fft + w].real();
                }
            }
        }

        if (kernel.requires_grad()) {
            std::vector<std::complex<double>> in_fft(fft_slice_size, {0.0, 0.0});
            const float* in_slice_ptr = a_data + b * in_slice_size;
            for (int64_t h = 0; h < H_in; ++h) {
                for (int64_t w = 0; w < W_in; ++w) {
                    in_fft[h * W_fft + w] = std::complex<double>(in_slice_ptr[h * W_in + w], 0.0);
                }
            }
            fft2d(in_fft, H_fft, W_fft);

            std::vector<std::complex<double>> kernel_grad_fft(fft_slice_size);
            for (int i = 0; i < fft_slice_size; ++i) {
                kernel_grad_fft[i] = out_grad_fft[i] * in_fft[i];
            }
            ifft2d(kernel_grad_fft, H_fft, W_fft);

            float* kernel_grad_slice_ptr = kernel_grad_data + b * kernel_slice_size;
            for (int64_t h = 0; h < H_k; ++h) {
                for (int64_t w = 0; w < W_k; ++w) {
                    kernel_grad_slice_ptr[h * W_k + w] += kernel_grad_fft[h * W_fft + w].real();
                }
            }
        }
    }
}
