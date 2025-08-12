#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include <vector>
#include <complex>
#include <stdexcept>
#include <omp.h>

void col2im(const float* data_col, const int64_t channels,
            const int64_t height, const int64_t width,
            const int64_t kernel_h, const int64_t kernel_w,
            const int64_t pad_h, const int64_t pad_w,
            const int64_t stride_h, const int64_t stride_w,
            float* data_im) {
    const int64_t output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int64_t output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    const int64_t channel_size = height * width;

    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
            for (int64_t kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {
                int64_t input_row_base = -pad_h + kernel_row;
                for (int64_t output_row = 0; output_row < output_h; ++output_row) {
                    int64_t input_row = input_row_base + output_row * stride_h;
                    if (is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        int64_t input_col_base = -pad_w + kernel_col;
                        for (int64_t output_col = 0; output_col < output_w; ++output_col) {
                            int64_t input_col = input_col_base + output_col * stride_w;
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                data_im[input_row * width + input_col] += *data_col;
                            }
                            data_col++;
                        }
                    } else {
                        data_col += output_w;
                    }
                }
            }
        }
        data_im += channel_size;
    }
}

void CpuAutograd::conv2d(Tensor& out, std::vector<Tensor>& prev, int stride, int padding) {
    if (prev.size() != 2) {
        throw std::runtime_error("conv2d_backward expects 2 previous tensors (input and kernel)");
    }

    Tensor t = out;
    Tensor& a = prev[0];
    Tensor& kernel = prev[1];

    if (!a.requires_grad() && !kernel.requires_grad()) {
        return;
    }

    const std::vector<int64_t>& a_shape = a.shape();
    const std::vector<int64_t>& kernel_shape = kernel.shape();
    const std::vector<int64_t>& out_grad_shape = out.shape();

    const int64_t batch_size = a_shape.size() > 3 ? a_shape[0] : 1;
    const int64_t C_in = a_shape.size() > 2 ? a_shape[a_shape.size() - 3] : 1;
    const int64_t H_in = a_shape[a_shape.size() - 2];
    const int64_t W_in = a_shape[a_shape.size() - 1];
    
    const int64_t C_out = kernel_shape[0];
    const int64_t H_k = kernel_shape[kernel_shape.size() - 2];
    const int64_t W_k = kernel_shape[kernel_shape.size() - 1];

    const int64_t H_out = out_grad_shape[out_grad_shape.size() - 2];
    const int64_t W_out = out_grad_shape[out_grad_shape.size() - 1];

    const float* a_data = static_cast<const float*>(a.data_ptr().get());
    const float* kernel_data = static_cast<const float*>(kernel.data_ptr().get());
    const float* out_grad_data = static_cast<const float*>(t.grad_ptr().get());

    float* a_grad_data = a.requires_grad() ? static_cast<float*>(a.grad_ptr().get()) : nullptr;
    float* kernel_grad_data = kernel.requires_grad() ? static_cast<float*>(kernel.grad_ptr().get()) : nullptr;

    const int64_t a_batch_stride = C_in * H_in * W_in;
    const int64_t out_grad_batch_stride = C_out * H_out * W_out;

    const int64_t kernel_matrix_rows = C_out;
    const int64_t kernel_matrix_cols = C_in * H_k * W_k;
    const int64_t col_buffer_cols = H_out * W_out;
    
    std::vector<float> kernel_grad_accumulator(kernel.numel(), 0.0f);

    #pragma omp parallel
    {
        std::vector<float> kernel_grad_private(kernel.numel(), 0.0f);

        #pragma omp for schedule(static)
        for (int64_t b = 0; b < batch_size; ++b) {
            const float* out_grad_batch = out_grad_data + b * out_grad_batch_stride;

            if (a.requires_grad()) {
                std::vector<float> a_grad_col(kernel_matrix_cols * col_buffer_cols);
                
                for (int64_t i = 0; i < kernel_matrix_cols; ++i) {
                    for (int64_t j = 0; j < col_buffer_cols; ++j) {
                        float sum = 0.0f;
                        for (int64_t k = 0; k < kernel_matrix_rows; ++k) {
                            sum += kernel_data[k * kernel_matrix_cols + i] * out_grad_batch[k * col_buffer_cols + j];
                        }
                        a_grad_col[i * col_buffer_cols + j] = sum;
                    }
                }

                float* a_grad_batch = a_grad_data + b * a_batch_stride;
                col2im(a_grad_col.data(), C_in, H_in, W_in, H_k, W_k, padding, padding, stride, stride, a_grad_batch);
            }

            if (kernel.requires_grad()) {
                const float* a_data_batch = a_data + b * a_batch_stride;
                
                std::vector<float> a_col(kernel_matrix_cols * col_buffer_cols);
                im2col(a_data_batch, C_in, H_in, W_in, H_k, W_k, padding, padding, stride, stride, a_col.data());

                for (int64_t i = 0; i < kernel_matrix_rows; ++i) {
                    for (int64_t j = 0; j < kernel_matrix_cols; ++j) {
                        float sum = 0.0f;
                        for (int64_t k = 0; k < col_buffer_cols; ++k) {
                           sum += out_grad_batch[i * col_buffer_cols + k] * a_col[j * col_buffer_cols + k]; // Note transpose on a_col
                        }
                        kernel_grad_private[i * kernel_matrix_cols + j] += sum;
                    }
                }
            }
        }

        #pragma omp critical
        {
            for(size_t i = 0; i < kernel.numel(); ++i) {
                kernel_grad_accumulator[i] += kernel_grad_private[i];
            }
        }
    }

    if (kernel.requires_grad()) {
        for (size_t i = 0; i < kernel.numel(); ++i) {
            kernel_grad_data[i] += kernel_grad_accumulator[i];
        }
    }
}


