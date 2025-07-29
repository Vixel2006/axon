#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/mean.h"

Tensor MeanImpl::forward_cpu(const Tensor& a, int dim, bool keepdim) {
    return mean_cpu(a, dim, keepdim);
}

Tensor MeanImpl::forward_gpu(const Tensor& a, int dim, bool keepdim) {
    return mean_gpu(a, dim, keepdim);
}

Tensor MeanImpl::backward_cpu(const Tensor& a) {
    throw std::runtime_error("Not implemented");
}

Tensor MeanImpl::backward_gpu(const Tensor& a) {
    throw std::runtime_error("Not implemented");
}


