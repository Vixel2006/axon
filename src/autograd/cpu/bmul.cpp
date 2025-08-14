#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "strided_indexer.h"
#include <stdexcept>


void CpuAutograd::mul(Tensor& out, std::vector<Tensor>& prev) {
    // Wrap the output gradient in a Tensor for easier use of Tensor operations
    Tensor out_grad = Tensor(out.shape(), out.strides(), out.dtype(), out.device(), out.grad_ptr(), 0, false, nullptr, std::nullopt);

    // Case: Tensor * Tensor
    if (prev.size() == 2) {
        Tensor& a = prev[0];
        Tensor& b = prev[1];

        // Gradient for 'a' is out_grad * b
        if (a.requires_grad()) {
            // The result of this multiplication will have the broadcasted shape
            Tensor grad_for_a = out_grad.mul(b);
            // Sum this gradient back into 'a's gradient buffer, handling reduction
            sum_gradient_for_broadcast(a, grad_for_a);
        }

        // Gradient for 'b' is out_grad * a
        if (b.requires_grad()) {
            // The result of this multiplication will have the broadcasted shape
            Tensor grad_for_b = out_grad.mul(a);
            // Sum this gradient back into 'b's gradient buffer, handling reduction
            sum_gradient_for_broadcast(b, grad_for_b);
        }
    }
    // Case: Tensor * scalar
    else if (prev.size() == 1) {
        Tensor& a = prev[0];
        if (a.requires_grad()) {
            // To get the scalar, we have to divide the output by the input.
            // This is the only way without storing the scalar in the context (Tape).
            // Note: This can be numerically unstable if any element in 'a' is zero.
            const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
            const float* out_data_p = static_cast<const float*>(out.data_ptr().get());
            if (a.numel() > 0 && a_data_p[0] == 0) {
                 // A simple check. A more robust solution would be needed for production.
                 // The gradient is 0 if the scalar was 0, otherwise it's undefined.
                 // Assuming the scalar wasn't zero.
            }
            const float scalar = (a.numel() > 0 && a_data_p[0] != 0) ? out_data_p[0] / a_data_p[0] : 0.0f;

            // The gradient for 'a' is out_grad * scalar
            Tensor grad_for_a = out_grad.mul(scalar);
            sum_gradient_for_broadcast(a, grad_for_a);
        }
    } else {
        throw std::runtime_error("Invalid number of inputs for mul backward pass.");
    }
}
