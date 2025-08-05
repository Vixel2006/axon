#include "tensor.h"
#include "engine/ops.h"
#include "helpers.h"
#include <numeric>
#include <vector>
#include <stdexcept>
#include <omp.h>

Tensor CpuOps::mean(const Tensor &a, int dim, bool keepdim) {
    int ndim = a.ndim();
    if (dim < 0) {
        dim = ndim + dim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Mean dimension is out of bounds.");
    }

    std::vector<int64_t> new_shape = reduce_shape(a.shape(), dim, keepdim);
    Tensor result = Tensor(new_shape, a.dtype(), deviceToString(a.device()), false);

    const int64_t reduction_size = a.shape()[dim];

    float* out_ptr = static_cast<float*>(result.data_ptr().get());

    if (reduction_size == 0) {
        const int64_t num_elements = result.numel();
        #pragma omp parallel for
        for (int64_t i = 0; i < num_elements; ++i) {
            out_ptr[i] = 0.0f;
        }
        return result;
    }

    const float* in_ptr = static_cast<const float*>(a.data_ptr().get());
    const auto& in_shape = a.shape();
    const auto& in_strides = a.strides();
    const int64_t reduction_stride = in_strides[dim];

    const float inv_reduction_size = 1.0f / static_cast<float>(reduction_size);

    if (reduction_stride == 1) {
        const int64_t outer_loop_size = result.numel();
        const int64_t inner_loop_size = reduction_size;

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < outer_loop_size; ++i) {
            const float* current_in_ptr = in_ptr + i * inner_loop_size;
            float local_sum = 0.0f;

            #pragma omp simd reduction(+:local_sum)
            for (int64_t j = 0; j < inner_loop_size; ++j) {
                local_sum += current_in_ptr[j];
            }
            out_ptr[i] = local_sum * inv_reduction_size;
        }
    }
    else {
        const int64_t outer_dims = std::accumulate(in_shape.begin(), in_shape.begin() + dim, 1LL, std::multiplies<int64_t>());
        const int64_t inner_dims = std::accumulate(in_shape.begin() + dim + 1, in_shape.end(), 1LL, std::multiplies<int64_t>());

        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 0; i < outer_dims; ++i) {
            for (int64_t j = 0; j < inner_dims; ++j) {
                const int64_t out_idx = i * inner_dims + j;
                const int64_t start_offset = (dim > 0 ? i * in_strides[dim - 1] : 0) + j;
                const float* in_base_ptr = in_ptr + start_offset;

                float local_sum = 0.0f;

                #pragma omp simd reduction(+:local_sum)
                for (int64_t k = 0; k < reduction_size; ++k) {
                    local_sum += in_base_ptr[k * reduction_stride];
                }
                out_ptr[out_idx] = local_sum * inv_reduction_size;
            }
        }
    }

    return result;
}

