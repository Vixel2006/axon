#include "tensor.h"
#include "engine/ops/impl/sum.h"
#include "helpers.h"

#include <numeric>

Tensor sum_cpu(const Tensor &a, int dim, bool keepdim) {
int ndim = a.ndim();
    if (dim < 0) {
        dim = ndim + dim;
    }

    std::vector<int64_t> new_shape = reduce_shape(a.shape(), dim, keepdim);
    Tensor result(new_shape, a.dtype(), "cpu");

    const float* in_ptr = static_cast<const float*>(a.raw_ptr()) + a.offset();
    float* out_ptr = static_cast<float*>(result.raw_ptr()) + result.offset();

    const auto& in_shape = a.shape();
    const auto& in_strides = a.strides();
    const auto& out_strides = compute_strides_(new_shape);

    const int64_t reduction_size = in_shape[dim];
    const int64_t reduction_stride = in_strides[dim];

    #pragma omp parallel for
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
            int64_t coord = temp_i / out_strides[out_dim_idx];
            start_in_offset += coord * in_strides[in_dim_idx];
            temp_i %= out_strides[out_dim_idx];

            out_dim_idx++;
        }

        float local_sum = 0.0f;
        for (int64_t j = 0; j < reduction_size; ++j) {
            local_sum += in_ptr[start_in_offset + j * reduction_stride];
        }
        out_ptr[i] = local_sum;
    }

    return result;
}

