#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/mul.h"

Tensor MulImpl::forward_cpu(const Tensor& a, const Tensor& b) {
    return mul_cpu(a, b);
}

Tensor MulImpl::forward_gpu(const Tensor& a, const Tensor& b) {
    return mul_gpu(a, b);
}

    Tensor MulImpl::backward_cpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}

Tensor MulImpl::backward_gpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}
