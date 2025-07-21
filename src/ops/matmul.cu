#include "tensor.h"
#include "ops/matmul.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <helpers.h>
#include <cstdio>

#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                     \
    }                                                                         \
} while (0)

#define TILE_DIM 32

Tensor matmul_gpu(const Tensor& a, const Tensor& b) {
    if (a.ndim() < 2 || b.ndim() < 2) { throw std::invalid_argument("Matmul requires at least 2 dimensions") }

    // We need to do broadcast on the batch dimension
    a_broadcasted = a.expand(a.shape().front(), b.shape().front());
    b_broadcasted = b.expand(a.shape().front(), b.shape().front());
}
