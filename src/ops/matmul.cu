#include "ops/matmul.h"
#include "tensor.h"
#include "utils.h"
#include "helpers.h"
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void batched_matmul_kernel(
    const float* a, const float* b, float* c,
    int64_t M, int64_t N, int64_t K,
    int64_t a_stride_m, int64_t a_stride_k,
    int64_t b_stride_k, int64_t b_stride_n,
    int64_t c_stride_m, int64_t c_stride_n,
    int64_t a_batch_stride, int64_t b_batch_stride, int64_t c_batch_stride
) {}

Tensor matmul_gpu(const Tensor& a, const Tensor& b) {}
