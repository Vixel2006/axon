#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "init.h"
#include <vector>
#include <complex>
#include <numeric>
#include <cmath>

void fft(std::vector<std::complex<double>>& x) {
    int N = x.size();
    if (N <= 1) return;
    std::vector<std::complex<double>> even(N/2), odd(N/2);
    for (int i = 0; i < N / 2; ++i) {
        even[i] = x[i*2];
        odd[i] = x[i*2 + 1];
    }
    fft(even);
    fft(odd);
    for (int k = 0; k < N / 2; ++k) {
        std::complex<double> t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

void ifft(std::vector<std::complex<double>>& x) {
    int N = x.size();
    for (auto& val : x) val = std::conj(val);
    fft(x);
    for (auto& val : x) val = std::conj(val) / static_cast<double>(N);
}

void fft2d(std::vector<std::complex<double>>& data, int H, int W) {
    std::vector<std::complex<double>> row(W);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) row[j] = data[i * W + j];
        fft(row);
        for (int j = 0; j < W; ++j) data[i * W + j] = row[j];
    }
    std::vector<std::complex<double>> col(H);
    for (int j = 0; j < W; ++j) {
        for (int i = 0; i < H; ++i) col[i] = data[i * W + j];
        fft(col);
        for (int i = 0; i < H; ++i) data[i * W + j] = col[i];
    }
}

void ifft2d(std::vector<std::complex<double>>& data, int H, int W) {
    std::vector<std::complex<double>> row(W);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) row[j] = data[i * W + j];
        ifft(row);
        for (int j = 0; j < W; ++j) data[i * W + j] = row[j];
    }
    std::vector<std::complex<double>> col(H);
    for (int j = 0; j < W; ++j) {
        for (int i = 0; i < H; ++i) col[i] = data[i * W + j];
        ifft(col);
        for (int i = 0; i < H; ++i) data[i * W + j] = col[i];
    }
}

Tensor CpuOps::conv2d(const Tensor& a, const Tensor& kernel, int stride, int padding) {
    const std::vector<int64_t>& in_shape = a.shape();
    const std::vector<int64_t>& kernel_shape = kernel.shape();

    const int64_t H_in = in_shape[in_shape.size() - 2];
    const int64_t W_in = in_shape[in_shape.size() - 1];
    const int64_t H_k = kernel_shape[kernel_shape.size() - 2];
    const int64_t W_k = kernel_shape[kernel_shape.size() - 1];

    const int64_t H_fft = H_in + H_k - 1;
    const int64_t W_fft = W_in + W_k - 1;

    int64_t batch_dims = 1;
    for (size_t i = 0; i < in_shape.size() - 2; ++i) {
        batch_dims *= in_shape[i];
    }

    std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end() - 2);
    out_shape.push_back(H_fft);
    out_shape.push_back(W_fft);
    Tensor out = zeros(out_shape, deviceToString(a.device()), a.requires_grad());

    float* a_data = static_cast<float*>(a.data_ptr().get());
    float* kernel_data = static_cast<float*>(kernel.data_ptr().get());
    float* out_data = static_cast<float*>(out.data_ptr().get());
    
    const int64_t in_slice_size = H_in * W_in;
    const int64_t kernel_slice_size = H_k * W_k;
    const int64_t fft_slice_size = H_fft * W_fft;

    for (int64_t b = 0; b < batch_dims; ++b) {
        std::vector<std::complex<double>> in_fft(fft_slice_size, {0.0, 0.0});
        std::vector<std::complex<double>> kernel_fft(fft_slice_size, {0.0, 0.0});

        float* in_slice_ptr = a_data + b * in_slice_size;
        float* kernel_slice_ptr = kernel_data + b * kernel_slice_size;
        
        for (int64_t h = 0; h < H_in; ++h) {
            for (int64_t w = 0; w < W_in; ++w) {
                in_fft[h * W_fft + w] = std::complex<double>(in_slice_ptr[h * W_in + w], 0.0);
            }
        }
        
        for (int64_t h = 0; h < H_k; ++h) {
            for (int64_t w = 0; w < W_k; ++w) {
                kernel_fft[h * W_fft + w] = std::complex<double>(kernel_slice_ptr[h * W_k + w], 0.0);
            }
        }
        
        fft2d(in_fft, H_fft, W_fft);
        fft2d(kernel_fft, H_fft, W_fft);
        
        for (int i = 0; i < fft_slice_size; ++i) {
            in_fft[i] *= kernel_fft[i];
        }

        ifft2d(in_fft, H_fft, W_fft);
        
        float* out_slice_ptr = out_data + b * fft_slice_size;
        for (int i = 0; i < fft_slice_size; ++i) {
            out_slice_ptr[i] = in_fft[i].real();
        }
    }

    return out;
}
