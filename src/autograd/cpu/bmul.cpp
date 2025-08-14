#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "strided_indexer.h"
#include <stdexcept>


void CpuAutograd::mul(Tensor& out, std::vector<Tensor>& prev) {
    Tensor out_grad = Tensor(out.shape(), out.strides(), out.dtype(), out.device(), out.grad_ptr(), 0, false, nullptr, std::nullopt);

    if (prev.size() == 2) {
        Tensor& a = prev[0];
        Tensor& b = prev[1];

        if (a.requires_grad()) {
            Tensor grad_for_a = out_grad.mul(b);
            sum_gradient_for_broadcast(a, grad_for_a);
        }

        if (b.requires_grad()) {
            Tensor grad_for_b = out_grad.mul(a);
            sum_gradient_for_broadcast(b, grad_for_b);
        }
    }
    else if (prev.size() == 1) {
        Tensor& a = prev[0];
        if (a.requires_grad()) {
            const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
            const float* out_data_p = static_cast<const float*>(out.data_ptr().get());
            if (a.numel() > 0 && a_data_p[0] == 0) {
            }
            const float scalar = (a.numel() > 0 && a_data_p[0] != 0) ? out_data_p[0] / a_data_p[0] : 0.0f;

            Tensor grad_for_a = out_grad.mul(scalar);
            sum_gradient_for_broadcast(a, grad_for_a);
        }
    } else {
        throw std::runtime_error("Invalid number of inputs for mul backward pass.");
    }
}
