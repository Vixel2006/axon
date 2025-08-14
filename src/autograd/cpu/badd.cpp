#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "strided_indexer.h"
#include <stdexcept>
#include <numeric>

#pragma omp declare simd
void CpuAutograd::add(Tensor& out, std::vector<Tensor>& prev) {
    Tensor out_grad = Tensor(out.shape(), out.strides(), out.dtype(), out.device(), out.grad_ptr(), 0, false, nullptr, std::nullopt);
    
    if (prev.size() == 2) {
        Tensor& a = prev[0];
        Tensor& b = prev[1];
        
        sum_gradient_for_broadcast(a, out_grad);
        sum_gradient_for_broadcast(b, out_grad);

    } else if (prev.size() == 1) {
        Tensor& a = prev[0];
        sum_gradient_for_broadcast(a, out_grad);
        
    } else {
        throw std::runtime_error("Invalid number of inputs for add backward pass.");
    }
}

