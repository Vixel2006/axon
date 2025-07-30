#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/add.h"


Tensor AddImpl::forward_cpu(const Tensor& a, const Tensor& b) {
    Tensor t = add_cpu(a, b);

    if (t.requires_grad()) {
      t.set_ctx({a, b}, '+');
    }
    return t;
}

Tensor AddImpl::forward_gpu(const Tensor& a, const Tensor& b) {
    Tensor t = add_gpu(a, b);

    if (t.requires_grad()) {
      std::vector<Tensor> prev;
      prev.push_back(a); prev.push_back(b);
      char op = '+';
      t.set_ctx(prev, op);
    }

    return t;
}

Tensor AddImpl::backward_cpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}

Tensor AddImpl::backward_gpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}
