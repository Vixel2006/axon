#include "engine/ops.h"
#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "init.h"
#include <vector>
#include <complex>
#include <numeric>
#include <cmath>
#include <omp.h>
#include <cstdint>

Tensor CpuOps::conv2d(const Tensor& a, const Tensor& kernel, int stride, int padding) {
    const std::vector<int64_t>& in_shape = a.shape();
    const std::vector<int64_t>& kernel_shape = kernel.shape();

    const int64_t batch_size = in_shape.size() > 3 ? in_shape[0] : 1;
    const int64_t C_in = in_shape.size() > 2 ? in_shape[in_shape.size() - 3] : 1;
    const int64_t H_in = in_shape[in_shape.size() - 2];
    const int64_t W_in = in_shape[in_shape.size() - 1];
    
    const int64_t C_out = kernel_shape[0];
    const int64_t H_k = kernel_shape[kernel_shape.size() - 2];
    const int64_t W_k = kernel_shape[kernel_shape.size() - 1];

    const int64_t H_out = (H_in + 2 * padding - H_k) / stride + 1;
    const int64_t W_out = (W_in + 2 * padding - W_k) / stride + 1;

    std::vector<int64_t> out_shape = {batch_size, C_out, H_out, W_out};
    Tensor out = zeros(out_shape, deviceToString(a.device()), a.requires_grad());

    float* a_data = static_cast<float*>(a.data_ptr().get());
    float* kernel_data = static_cast<float*>(kernel.data_ptr().get());
    float* out_data = static_cast<float*>(out.data_ptr().get());
    
    const int64_t in_batch_stride = C_in * H_in * W_in;
    const int64_t out_batch_stride = C_out * H_out * W_out;
    const int64_t out_channel_size = H_out * W_out;

    const int64_t kernel_matrix_rows = C_out;
    const int64_t kernel_matrix_cols = C_in * H_k * W_k;
    const int64_t col_buffer_rows = kernel_matrix_cols;
    const int64_t col_buffer_cols = H_out * W_out;

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < batch_size; ++b) {
        std::vector<float> col_buffer(col_buffer_rows * col_buffer_cols, 0.0f);
        float* col_buffer_data = col_buffer.data();
        
        const float* a_data_batch = a_data + b * in_batch_stride;
        
        // Fixed im2col call
        im2col(a_data_batch, C_in, H_in, W_in, H_k, W_k, padding, padding, stride, stride, col_buffer_data);
        
        float* out_data_batch = out_data + b * out_batch_stride;
        
        // Matrix multiplication: kernel (C_out x (C_in*H_k*W_k)) * col_buffer ((C_in*H_k*W_k) x (H_out*W_out))
        for (int64_t i = 0; i < kernel_matrix_rows; ++i) {
            for (int64_t j = 0; j < col_buffer_cols; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < kernel_matrix_cols; ++k) {
                    sum += kernel_data[i * kernel_matrix_cols + k] * col_buffer_data[k * col_buffer_cols + j];
                }
                out_data_batch[i * col_buffer_cols + j] = sum;
            }
        }
    }

    if (out.requires_grad()) {
      auto backward_fn = [stride, padding](Tensor& out, std::vector<Tensor>& prev) {
        CpuAutograd::conv2d(out, prev, stride, padding);
      };
      out.set_ctx({a, kernel}, backward_fn);
    }

    return out;
}
