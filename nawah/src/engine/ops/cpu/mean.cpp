#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops.h"
#include <stdexcept>
#include <omp.h>

Tensor CpuOps::mean(const Tensor &a, int dim, bool keepdim) {
    int ndim = a.ndim();
    if (dim < 0) {
        dim = ndim + dim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Mean dimension out of range.");
    }

    const int64_t reduction_size = a.shape()[dim];

    Tensor sum_result = CpuOps::sum(a, dim, keepdim);

    if (reduction_size == 0) {
        return sum_result;
    }

    float* result_ptr = static_cast<float*>(sum_result.raw_ptr()) + sum_result.offset();
    const int64_t num_elements = sum_result.numel();

    #pragma omp parallel for
    for (int64_t i = 0; i < num_elements; ++i) {
        result_ptr[i] /= static_cast<float>(reduction_size);
    }

    return sum_result;
}
