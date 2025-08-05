#include "tensor.h"
#include "engine/ops.h"
#include "helpers.h"
#include <numeric>
#include <vector>
#include <stdexcept>
#include <omp.h>

Tensor CpuOps::sum(const Tensor &a, int dim, bool keepdim) {
    int ndim = a.ndim();
    if (dim < 0) {
        dim = ndim + dim;
    }

    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Reduction dimension is out of bounds.");
    }

    std::vector<int64_t> new_shape = reduce_shape(a.shape(), dim, keepdim);
    Tensor result(new_shape, a.dtype(), deviceToString(a.device()), false);

    const float* in_ptr = static_cast<const float*>(a.data_ptr().get());
    float* out_ptr = static_cast<float*>(result.data_ptr().get());

    const auto& in_shape = a.shape();
    const auto& in_strides = a.strides();
    const int64_t reduction_size = in_shape[dim];
    const int64_t reduction_stride = in_strides[dim];

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
            out_ptr[i] = local_sum;
        }
    }
    else {
        const auto& out_strides = result.strides();

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < result.numel(); ++i) {
            int64_t start_in_offset = 0;
            int64_t temp_i = i;

            int out_dim_idx = 0;
            for (int in_dim_idx = 0; in_dim_idx < ndim; ++in_dim_idx) {
                if (in_dim_idx == dim) {
                    if (keepdim) {
                        out_dim_idx++;
                    }
                    continue;
                }

                const int64_t coord = temp_i / out_strides[out_dim_idx];
                start_in_offset += coord * in_strides[in_dim_idx];
                temp_i %= out_strides[out_dim_idx];

                out_dim_idx++;
            }

            float local_sum = 0.0f;
            #pragma omp simd reduction(+:local_sum)
            for (int64_t j = 0; j < reduction_size; ++j) {
                local_sum += in_ptr[start_in_offset + j * reduction_stride];
            }
            out_ptr[i] = local_sum;
        }
    }

    return result;
}
