#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/sub.h"

Tensor SubImpl::forward_cpu(const Tensor& a, const Tensor& b) {
    Tensor t = sub_cpu(a, b);

    if (t.requires_grad()) {
      std::vector<Tensor> prev;
      prev.push_back(a); prev.push_back(b);
      char op = '-';
      t.set_ctx(prev, op);
    }

    return t;
}

Tensor SubImpl::forward_gpu(const Tensor& a, const Tensor& b) {
    Tensor t = sub_gpu(a, b);

    if (t.requires_grad()) {
      std::vector<Tensor> prev;
      prev.push_back(a); prev.push_back(b);
      char op = '-';
      t.set_ctx(prev, op);
    }

    return t;
}

Tensor SubImpl::backward_cpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}

Tensor SubImpl::backward_gpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}
