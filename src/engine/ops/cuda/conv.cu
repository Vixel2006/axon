#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include "autograd/ops.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

__global__ void conv2d_kernel(const float* a_data, const float* k_data, float* res_data,
                                      int N, int C_in, int H_in, int W_in,
                                      int C_out, int KH, int KW,
                                      int stride, int padding,
                                      int H_out, int W_out,
                                      const int64_t* a_strides, const int64_t* k_strides, const int64_t* res_strides) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z % C_out;
    int n = blockIdx.z / C_out;

    if (w_out >= W_out || h_out >= H_out || n >= N) {
        return;
    }

    float acc = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int h_in = h_out * stride + kh - padding;
                int w_in = w_out * stride + kw - padding;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int64_t a_idx = n * a_strides[0] + c_in * a_strides[1] + h_in * a_strides[2] + w_in * a_strides[3];
                    int64_t k_idx = c_out * k_strides[0] + c_in * k_strides[1] + kh * k_strides[2] + kw * k_strides[3];
                    acc += a_data[a_idx] * k_data[k_idx];
                }
            }
        }
    }

    int64_t res_idx = n * res_strides[0] + c_out * res_strides[1] + h_out * res_strides[2] + w_out * res_strides[3];
    res_data[res_idx] = acc;
}

Tensor CudaOps::conv2d(const Tensor& a, const Tensor& kernel, int stride, int padding) {
}

