#include "tensor.h"
#include "engine/ops/impl/matmul.h"
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
    // TODO: Implement matmul_gpu
}
