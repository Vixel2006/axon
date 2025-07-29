#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/matmul.h"

Tensor MatmulImpl::forward_cpu(const Tensor& a, const Tensor& b) {
    return matmul_cpu(a, b);
}

Tensor MatmulImpl::forward_gpu(const Tensor& a, const Tensor& b) {
    return matmul_gpu(a, b);
}

Tensor MatmulImpl::backward_cpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}

Tensor MatmulImpl::backward_gpu(const Tensor& a, const Tensor& b) {
        throw std::runtime_error("Not implemented");
}
