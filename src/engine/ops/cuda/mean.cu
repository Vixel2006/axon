#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA Error in " #call " : ") + \
                                     cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

struct AlignedDeleter {
    void operator()(void* ptr) const {
        #ifdef _MSC_VER
        _aligned_free(ptr);
        #else
        free(ptr);
        #endif
    }
};

__global__ void sum_reduce_kernel_for_mean(const float* input_data, float* output_data, size_t num_elements) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    float my_sum = 0.0f;
    for (size_t j = i; j < num_elements; j += stride) {
        my_sum += input_data[j];
    }
    sdata[tid] = my_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_data[blockIdx.x] = sdata[0];
    }
}

Tensor CudaOps::mean(const Tensor &a, int axis, bool keep_dims) {
}

