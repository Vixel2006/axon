#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/matmul.h"

Tensor MatmulImpl::forward_cpu(const Tensor& a, const Tensor& b) {
    Tensor t = matmul_cpu(a, b);

    if (t.requires_grad()) {
      std::vector<Tensor> prev;
      prev.push_back(a); prev.push_back(b);
      char op = '@';
      t.set_ctx(prev, op);
    }

    return t;
}

Tensor MatmulImpl::forward_gpu(const Tensor& a, const Tensor& b) {
    Tensor t = matmul_gpu(a, b);

    if (t.requires_grad()) {
      std::vector<Tensor> prev; 
      prev.push_back(a); prev.push_back(b);
      char op = '@';
      t.set_ctx(prev, op);
    }

    return t;
}

Tensor MatmulImpl::backward_cpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}

Tensor MatmulImpl::backward_gpu(const Tensor& a, const Tensor& b) {
        throw std::runtime_error("Not implemented");
}
