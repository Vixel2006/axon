#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/sum.h"

Tensor SumImpl::forward_cpu(const Tensor& a, int dim, bool keepdim) {
    return sum_cpu(a, dim, keepdim);
}

Tensor SumImpl::forward_gpu(const Tensor& a, int dim, bool keepdim) {
    return sum_gpu(a, dim, keepdim);
}

Tensor SumImpl::backward_cpu(const Tensor& a) {
    throw std::runtime_error("Not implemented");
}

Tensor SumImpl::backward_gpu(const Tensor& a) {
    throw std::runtime_error("Not implemented");
}

