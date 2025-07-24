#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/add.h"


Tensor AddImpl::forward_cpu(const Tensor& a, const Tensor& b) {
    return add_cpu(a, b);
}

Tensor AddImpl::forward_gpu(const Tensor& a, const Tensor& b) {
    return add_gpu(a, b);
}

Tensor AddImpl::backward_cpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}

Tensor AddImpl::backward_gpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}
