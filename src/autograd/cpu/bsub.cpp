#include "tensor.h"
#include "autograd/ops.h"
#include "helpers.h"
#include "strided_indexer.h"
#include <stdexcept>


void CpuAutograd::sub(Tensor& out, std::vector<Tensor>& prev) {
    // Wrap the output gradient in a Tensor for easier handling of shape/strides
    Tensor out_grad = Tensor(out.shape(), out.strides(), out.dtype(), out.device(), out.grad_ptr(), 0, false, nullptr, std::nullopt);

    // Case for Tensor - Tensor
    if (prev.size() == 2) {
        Tensor& a = prev[0];
        Tensor& b = prev[1];

        // For input 'a', the gradient is +out_grad (da/da = 1)
        if (a.requires_grad()) {
            sum_gradient_for_broadcast(a, out_grad);
        }

        // For input 'b', the gradient is -out_grad (da/db = -1)
        if (b.requires_grad()) {
            // Create a temporary tensor containing the negated output gradient
            Tensor neg_out_grad = out_grad.neg();
            // Accumulate the negated gradient into b
            sum_gradient_for_broadcast(b, neg_out_grad);
        }
    }
    // Case for Tensor - scalar
    else if (prev.size() == 1) {
        Tensor& a = prev[0];
        // The gradient only flows back to the tensor, not the scalar
        if (a.requires_grad()) {
            sum_gradient_for_broadcast(a, out_grad);
        }
    }
    else {
        throw std::runtime_error("Invalid number of inputs for sub backward pass.");
    }
}


