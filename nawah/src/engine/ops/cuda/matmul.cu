#include "tensor.h"
#include "engine/ops.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include "helpers.h"
#include "utils.h"
#include <cstdio>
#include <cstdio>

// Helper to convert cuBLAS status to a string


#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                     \
    }                                                                         \
} while (0)

#define TILE_DIM 32

Tensor CudaOps::matmul(const Tensor& a, const Tensor& b) {
  // Won't do it now
}

