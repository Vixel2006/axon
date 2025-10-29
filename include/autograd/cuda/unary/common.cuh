#ifndef AUTOGRAD_CUDA_UNARY_COMMON_CUH
#define AUTOGRAD_CUDA_UNARY_COMMON_CUH

#include "autograd/autograd_unary.h"
#include "cuda_utils.h" // For CHECK_CUDA
#include "logger.h"
#include "tensor.h"
#include "utils.h" // For numel
#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>

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
__global__ void relu_grad_kernel(const float* out_grad, const float* prev_data, float* prev_grad,
                                 int n);
__global__ void noncontig_relu_grad_kernel(const float* out_grad, const float* prev_data,
                                          float* prev_grad, int n, const int* shape,
                                          const int* strides, int ndim);
__global__ void noncontig_abs_grad_kernel(const float* out_grad, const float* prev_data,
                                          float* prev_grad, int n, const int* shape,
                                          const int* strides, int ndim);
__global__ void log_grad_kernel(const float* out_grad, const float* prev_data, float* prev_grad,
                                int n);
__global__ void noncontig_log_grad_kernel(const float* out_grad, const float* prev_data,
                                          float* prev_grad, int n, const int* shape,
                                          const int* strides, int ndim);
__global__ void exp_grad_kernel(const float* out_grad, const float* out_data, float* prev_grad,
                                int n);
__global__ void noncontig_exp_grad_kernel(const float* out_grad, const float* out_data,
                                          float* prev_grad, int n, const int* shape,
                                          const int* strides, int ndim);
__global__ void abs_grad_kernel(const float* out_grad, const float* prev_data, float* prev_grad,
                                int n);
__global__ void neg_grad_kernel(const float* out_grad, float* prev_grad, int n);
__global__ void noncontig_neg_grad_kernel(const float* out_grad, float* prev_grad, int n,
                                          const int* shape, const int* strides, int ndim);
__global__ void clip_grad_kernel(const float* out_grad, const float* prev_data, float* prev_grad,
                                 float min_val, float max_val, int n);
__global__ void noncontig_clip_grad_kernel(const float* out_grad, const float* prev_data,
                                           float* prev_grad, float min_val, float max_val, int n,
                                           const int* shape, const int* strides, int ndim);

#endif // AUTOGRAD_CUDA_UNARY_COMMON_CUH
