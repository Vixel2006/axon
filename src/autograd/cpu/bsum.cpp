#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include <vector>
#include <stdexcept>
#include <omp.h>

/**
 * @brief Backward pass for the sum reduction on the CPU, using in-place accumulation.
 *
 * This function propagates the gradient from the output of the sum operation
 * back to its input. The core logic is a broadcast: the gradient from an
 * element in the output is *added* to all the input elements that were
_summed
 * to create it.
 *
 * @param out The output tensor from the forward `sum` operation. Its gradient
 *            is the source for the backward pass.
 * @param prev A vector containing the single input tensor to the `sum` operation.
 *             Its gradient will be accumulated into.
 */
void CpuAutograd::sum(const Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    Tensor& a = prev[0];

    if (!a.requires_grad()) {
        return;
    }

    const float* grad_out_ptr = static_cast<const float*>(t.grad_ptr().get());
    float* grad_a_ptr = static_cast<float*>(a.grad_ptr().get());

    if (!grad_out_ptr || !grad_a_ptr) {
        throw std::runtime_error("A gradient pointer is null in 'sum' backward pass.");
    }

    const auto& in_shape = a.shape();
    const auto& out_shape = out.shape();
    const auto& in_strides = a.strides();
    const auto& out_strides = out.strides();
    const int ndim = a.ndim();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < a.numel(); ++i) {
        int64_t out_idx = 0;
        int64_t temp_i = i;

        for (int dim_idx = 0; dim_idx < ndim; ++dim_idx) {
            const int64_t coord = temp_i / in_strides[dim_idx];
            temp_i %= in_strides[dim_idx];

            if (dim_idx < out_shape.size() && out_shape[dim_idx] > 1) {
                out_idx += coord * out_strides[dim_idx];
            }
        }

        grad_a_ptr[i] += grad_out_ptr[out_idx];
    }
}
