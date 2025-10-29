#ifndef AUTOGRAD_CUDA_BINARY_COMMON_CUH
#define AUTOGRAD_CUDA_BINARY_COMMON_CUH

#include "autograd/autograd_binary.h"
#include "cuda_utils.h" // For CHECK_CUDA
#include "logger.h"
#include "ops/movement_ops.h" // This might be needed for matmul_grad_op_cuda
#include "tensor.h"
#include "utils.h" // For numel
#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_DIM 16

#define CHECK_CUDA()                                                                               \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("CUDA runtime error at %s:%d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
            assert(0 && "CUDA runtime error");                                                     \
        }                                                                                          \
    } while (0)

// Kernel declarations
__global__ void contig_add_grad_kernel(const float* out_grad, float* prev_grad, int n);

__global__ void noncontig_add_grad_kernel(const float* out_grad, float* prev_grad, int n,
                                          const int* shape, const int* strides, int ndim);
__global__ void noncontig_sub_grad_kernel(const float* out_grad, float* prev_grad, int n,
                                          const int* shape, const int* strides, int ndim);
__global__ void sub_grad_kernel(const float* out_grad, float* prev_grad, int n);
__global__ void mul_grad_kernel(const float* out_grad, float* prev_grad, float* other_data, int n);
__global__ void scalar_mul_grad_kernel(const float* out_grad, float* prev_grad, float scalar,
                                       int n);
__global__ void noncontig_mul_grad_kernel(const float* out_grad, float* prev_grad,
                                          float* other_data, int n, const int* shape,
                                          const int* strides, int ndim);
__global__ void noncontig_scalar_mul_grad_kernel(const float* out_grad, float* prev_grad,
                                                  float scalar, int n, const int* shape,
                                                  const int* strides, int ndim);
__global__ void scalar_pow_grad_kernel(const float* out_grad, float* prev_data, float* prev_grad,
                                       float power, int n);
__global__ void base_pow_grad_kernel(const float* out_grad, float* base_data, float* base_grad,
                                     float* power_data, float* power_grad, int n);
__global__ void exponent_pow_grad_kernel(const float* out_grad, const float* out_data,
                                         float* base_data, float* power_grad, int n);
__global__ void noncontig_scalar_pow_grad_kernel(const float* out_grad, float* prev_data,
                                                 float* prev_grad, float power, int n,
                                                 const int* shape, const int* strides, int ndim);
__global__ void noncontig_base_pow_grad_kernel(const float* out_grad, float* base_data,
                                               float* base_grad, float* power_data, int n,
                                               const int* shape, const int* strides, int ndim);
__global__ void noncontig_exponent_pow_grad_kernel(const float* out_grad, const float* out_data,
                                                   float* base_data, float* power_grad, int n,
                                                   const int* shape, const int* strides, int ndim);
__global__ void numerator_div_grad_kernel(const float* out_grad, float* prev_grad,
                                          const float* denominator, int n);
__global__ void denominator_div_grad_kernel(const float* out_grad, const float* out_data,
                                            float* prev_grad, float* denominator, int n);
__global__ void noncontig_numerator_div_grad_kernel(const float* out_grad, float* prev_grad,
                                                     const float* denominator, int n,
                                                     const int* shape, const int* strides, int ndim);
__global__ void noncontig_denominator_div_grad_kernel(const float* out_grad, const float* out_data,
                                                      float* prev_grad, float* denominator, int n,
                                                      const int* shape, const int* strides,
                                                      int ndim);
__global__ void scalar_div_grad_kernel(const float* out_grad, float* prev_grad,
                                       float scalar_denominator, int n);
__global__ void scalar_rdiv_grad_kernel(const float* out_grad, const float* out_data,
                                        float* prev_grad, float scalar_numerator,
                                        const float* prev_data, int n);
__global__ void noncontig_scalar_rdiv_grad_kernel(const float* out_grad, const float* out_data,
                                                  float* prev_grad, float scalar_numerator,
                                                  const float* prev_data, int n,
                                                  const int* shape, const int* strides, int ndim);
__global__ void matmul_grad_kernel(const float* lhs, const float* rhs, float* grad, int B, int N,
                                   int P, int K, bool transpose_lhs, bool transpose_rhs,
                                   bool is_lhs_batched, bool is_rhs_batched, bool is_grad_batched);

#endif // AUTOGRAD_CUDA_BINARY_COMMON_CUH
