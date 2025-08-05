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


Tensor CudaOps::sum(const Tensor &a, int dim, bool keepdim) {
}
