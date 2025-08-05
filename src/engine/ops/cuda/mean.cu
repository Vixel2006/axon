#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h" // For AlignedDeleter
#include <cuda_runtime.h>
#include <stdexcept>

// Macro for robust CUDA error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA Error in " #call " : ") + \
                                     cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

// A custom deleter for host memory allocated with posix_memalign or _aligned_malloc
struct AlignedDeleter {
    void operator()(void* ptr) const {
        #ifdef _MSC_VER
        _aligned_free(ptr);
        #else
        free(ptr);
        #endif
    }
};

/**
 * @brief A generic and scalable CUDA kernel for sum reduction.
 *
 * This kernel is used to calculate the sum, which is the first step of calculating the mean.
 * It can be used for both stages of a two-stage reduction.
 * 1. Each thread computes a partial sum over the input data using a grid-stride loop.
 * 2. The partial sums within a block are then reduced using shared memory.
 * 3. The thread with ID 0 in each block writes the block's final sum to the output array.
 *
 * @param input_data Pointer to the device memory of the input array.
 * @param output_data Pointer to the device memory of the output array (for partial sums).
 * @param num_elements The total number of elements in the input_data array.
 */
__global__ void sum_reduce_kernel_for_mean(const float* input_data, float* output_data, size_t num_elements) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    // Each thread performs a partial reduction from global memory
    float my_sum = 0.0f;
    for (size_t j = i; j < num_elements; j += stride) {
        my_sum += input_data[j];
    }
    sdata[tid] = my_sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output_data[blockIdx.x] = sdata[0];
    }
}

Tensor CudaOps::mean(const Tensor &a, int axis, bool keep_dims) {
}

