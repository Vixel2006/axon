#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include "autograd/ops.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <vector>

int next_power_of_2(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
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

Tensor CudaOps::conv2d(const Tensor& a, const Tensor& kernel, int stride, int padding) {
  if (stride != 1) {
    throw std::runtime_error("FFT-based conv2d only supports stride = 1.");
  }
  
  std::vector<int64_t> a_shape = a.shape();
  std::vector<int64_t> kernel_shape = kernel.shape();

  const int H_in = a_shape[a_shape.size() - 2];
  const int W_in = a_shape[a_shape.size() - 1];
  const int H_k = kernel_shape[kernel_shape.size() - 2];
  const int W_k = kernel_shape[kernel_shape.size() - 1];

  const int H_fft = next_power_of_2(H_in + H_k - 1);
  const int W_fft = next_power_of_2(W_in + W_k - 1);

  float *d_a, *d_kernel, *d_out;
  cufftComplex *d_a_fft, *d_k_fft;
  
  const size_t real_size = W_fft * H_fft * sizeof(float);
  const int W_fft_complex = (W_fft / 2) + 1;
  const size_t complex_size = W_fft_complex * H_fft * sizeof(cufftComplex);

  CUDA_CHECK(cudaMalloc(&d_a, real_size));
  CUDA_CHECK(cudaMalloc(&d_kernel, real_size));
  CUDA_CHECK(cudaMalloc(&d_a_fft, complex_size));
  CUDA_CHECK(cudaMalloc(&d_k_fft, complex_size));

  dim3 block_dim(16, 16);
  dim3 grid_dim((W_fft + block_dim.x - 1) / block_dim.x, (H_fft + block_dim.y - 1) / block_dim.y);
  
  pad_kernel<<<grid_dim, block_dim>>>(static_cast<float*>(a.data_ptr().get()), d_a, W_in, H_in, W_fft, H_fft);
  pad_kernel<<<grid_dim, block_dim>>>(static_cast<float*>(kernel.data_ptr().get()), d_kernel, W_k, H_k, W_fft, H_fft);

  cufftHandle plan_forward, plan_inverse;
  checkCufftErrors(cufftPlan2d(&plan_forward, H_fft, W_fft, CUFFT_R2C));
  checkCufftErrors(cufftPlan2d(&plan_inverse, H_fft, W_fft, CUFFT_C2R));
  
  checkCufftErrors(cufftExecR2C(plan_forward, (cufftReal*)d_a, d_a_fft));
  checkCufftErrors(cufftExecR2C(plan_forward, (cufftReal*)d_kernel, d_k_fft));
  
  const int complex_elements = W_fft_complex * H_fft;
  const float scale = 1.0f / (W_fft * H_fft);
  int threads_per_block_mult = 256;
  int blocks_mult = (complex_elements + threads_per_block_mult - 1) / threads_per_block_mult;

  complex_mult_and_scale_kernel<<<blocks_mult, threads_per_block_mult>>>(d_a_fft, d_k_fft, complex_elements, scale);

  checkCufftErrors(cufftExecC2R(plan_inverse, d_a_fft, (cufftReal*)d_a));
  
  const int H_out = (H_in + 2 * padding - H_k) / stride + 1;
  const int W_out = (W_in + 2 * padding - W_k) / stride + 1;
  
  std::vector<int64_t> out_shape;
  if (a_shape.size() > 2) {
      out_shape.insert(out_shape.end(), a_shape.begin(), a_shape.end() - 2);
  }
  out_shape.push_back(H_out);
  out_shape.push_back(W_out);
  
  Tensor out(out_shape, a.dtype(), deviceToString(a.device()), a.requires_grad());
  d_out = static_cast<float*>(out.data_ptr().get());

  CUDA_CHECK(cudaMemcpy2D(d_out, W_out * sizeof(float),
                                d_a, W_fft * sizeof(float),
                                W_out * sizeof(float), H_out,
                                cudaMemcpyDeviceToDevice));
  
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_kernel));
  CUDA_CHECK(cudaFree(d_a_fft));
  CUDA_CHECK(cudaFree(d_k_fft));
  checkCufftErrors(cufftDestroy(plan_forward));
  checkCufftErrors(cufftDestroy(plan_inverse));
  
  return out;
}
