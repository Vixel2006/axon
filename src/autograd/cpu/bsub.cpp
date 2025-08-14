#include "tensor.h"
#include "autograd/ops.h"
#include "helpers.h"
#include "strided_indexer.h"
#include <stdexcept>


void CpuAutograd::sub(Tensor& out, std::vector<Tensor>& prev) {
    Tensor out_grad = Tensor(out.shape(), out.strides(), out.dtype(), out.device(), out.grad_ptr(), 0, false, nullptr, std::nullopt);

    if (prev.size() == 2) {
        Tensor& a = prev[0];
        Tensor& b = prev[1];

        if (a.requires_grad()) {
            sum_gradient_for_broadcast(a, out_grad);
        }

        if (b.requires_grad()) {
            Tensor neg_out_grad = out_grad.neg();
            sum_gradient_for_broadcast(b, neg_out_grad);
        }
    }
    else if (prev.size() == 1) {
        Tensor& a = prev[0];
        if (a.requires_grad()) {
            sum_gradient_for_broadcast(a, out_grad);
        }
    }
    else {
        throw std::runtime_error("Invalid number of inputs for sub backward pass.");
    }
}


