#include <cuda_runtime.h>
#include "tensor.h"
#include "utils.h"
#include <cublas_v2.h>

__global__ void pad_grad_kernel(const float* out_grad, float* padded_grad,
                              const int W_out, const int H_out,
                              const int W_fft, const int H_fft,
                              const int W_k, const int H_k,
                              const int stride, const int padding) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;

    if (w_out < W_out && h_out < H_out) {
        int w_dst = w_out * stride + (W_k - 1) - padding;
        int h_dst = h_out * stride + (H_k - 1) - padding;

        if (w_dst >= 0 && w_dst < W_fft && h_dst >= 0 && h_dst < H_fft) {
            int src_idx = h_out * W_out + w_out;
            int dst_idx = h_dst * W_fft + w_dst;
            padded_grad[dst_idx] = out_grad[src_idx];
        }
    }
}

__global__ void pad_and_rotate_kernel(const float* kernel, float* padded_rotated_kernel,
                                    const int W_k, const int H_k,
                                    const int W_fft, const int H_fft) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W_k && y < H_k) {
        int src_idx = y * W_k + x;
        int dst_idx = (H_k - 1 - y) * W_fft + (W_k - 1 - x);
        padded_rotated_kernel[dst_idx] = kernel[src_idx];
    }
}

__global__ void crop_and_add_kernel(const float* full_conv_result, float* grad_tensor,
                                  const int W_crop, const int H_crop,
                                  const int W_fft) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W_crop && y < H_crop) {
        int src_idx = y * W_fft + x;
        int dst_idx = y * W_crop + x;
        atomicAdd(&grad_tensor[dst_idx], full_conv_result[src_idx]);
    }
}

__global__ void complex_mult_and_scale_kernel(cufftComplex* a, const cufftComplex* b, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ar = a[idx].x;
        float ai = a[idx].y;
        float br = b[idx].x;
        float bi = b[idx].y;
        a[idx].x = (ar * br - ai * bi) * scale;
        a[idx].y = (ar * bi + ai * br) * scale;
    }
}


__global__ void pad_kernel(const float* input, float* padded_output,
                         const int W_in, const int H_in,
                         const int W_padded, const int H_padded) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W_padded && y < H_padded) {
        int padded_idx = y * W_padded + x;
        if (x < W_in && y < H_in) {
            int input_idx = y * W_in + x;
            padded_output[padded_idx] = input[input_idx];
        } else {
            padded_output[padded_idx] = 0.0f;
        }
    }
}

__global__ void crop_and_stride_kernel(const float* full_conv_result, float* output, 
                                     const int W_full, const int W_out, const int H_out, 
                                     const int W_k, const int H_k, 
                                     const int stride, const int padding) {
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_out >= W_out || y_out >= H_out) {
        return;
    }

    int x_src = (x_out * stride - padding) + (W_k - 1);
    int y_src = (y_out * stride - padding) + (H_k - 1);

    int dst_idx = y_out * W_out + x_out;
    int src_idx = y_src * W_full + x_src;

    output[dst_idx] = full_conv_result[src_idx];
}

__global__ void im2col_kernel(const float* data_im, float* data_col,
                                        const int C_in, const int H_in, const int W_in,
                                        const int H_k, const int W_k,
                                        const int H_out, const int W_out,
                                        const int stride, const int padding) {
    for (int l = blockIdx.x * blockDim.x + threadIdx.x;
         l < (H_out * W_out);
         l += blockDim.x * gridDim.x) {

        const int w_out = l % W_out;
        const int h_out = l / W_out;
        const int w_in_start = w_out * stride - padding;
        const int h_in_start = h_out * stride - padding;

        for (int c = 0; c < C_in; ++c) {
            for (int kh = 0; kh < H_k; ++kh) {
                for (int kw = 0; kw < W_k; ++kw) {
                    const int h_in = h_in_start + kh;
                    const int w_in = w_in_start + kw;

                    const int k = (c * H_k + kh) * W_k + kw;
                    
                    const int idx_col = k * (H_out * W_out) + l;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        const int idx_im = (c * H_in + h_in) * W_in + w_in;
                        data_col[idx_col] = data_im[idx_im];
                    } else {
                        data_col[idx_col] = 0.0f;
                    }
                }
            }
        }
    }
}

__global__ void col2im_kernel(const float* data_col, float* data_im,
                              const int C_in, const int H_in, const int W_in,
                              const int H_k, const int W_k,
                              const int H_out, const int W_out,
                              const int stride, const int padding) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < (C_in * H_k * W_k * H_out * W_out);
         idx += blockDim.x * gridDim.x) {
        
        const int w_out = idx % W_out;
        const int h_out = (idx / W_out) % H_out;
        const int w_k = (idx / (W_out * H_out)) % W_k;
        const int h_k = (idx / (W_out * H_out * W_k)) % H_k;
        const int c_in = (idx / (W_out * H_out * W_k * H_k));

        const int h_in = h_out * stride - padding + h_k;
        const int w_in = w_out * stride - padding + w_k;
        
        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            atomicAdd(&data_im[(c_in * H_in + h_in) * W_in + w_in], data_col[idx]);
        }
    }
}

