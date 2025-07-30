#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/mul.h"

Tensor MulImpl::forward_cpu(const Tensor& a, const Tensor& b) {
    Tensor t = mul_cpu(a, b);

    if (t.requires_grad()) {
      std::vector<Tensor> prev;
      prev.push_back(a); prev.push_back(b);
      char op = '*';
      t.set_ctx(prev, op);
    }

    return t;
}

Tensor MulImpl::forward_gpu(const Tensor& a, const Tensor& b) {
    Tensor t = mul_gpu(a, b);

    if (t.requires_grad()) {
      std::vector<Tensor> prev;
      prev.push_back(a); prev.push_back(b);
      char op = '*';
      t.set_ctx(prev, op);
    }

    return t;
}

    Tensor MulImpl::backward_cpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}

Tensor MulImpl::backward_gpu(const Tensor& a, const Tensor& b) {
    throw std::runtime_error("Not implemented");
}
