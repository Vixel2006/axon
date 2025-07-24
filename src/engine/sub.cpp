#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/sub.h"

Tensor SubImpl::forward_cpu(const Tensor& a, const Tensor& b) {
    return sub_cpu(a, b);
}

Tensor SubImpl::forward_gpu(const Tensor& a, const Tensor& b) {
    return sub_gpu(a, b);
}

Tensor SubImpl::backward_cpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}

Tensor SubImpl::backward_gpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}
