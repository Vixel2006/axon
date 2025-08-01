#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/add.h"
#include "autograd/badd.h"

#include <vector>


Tensor AddImpl::cpu(const Tensor& a, const Tensor& b) {
    Tensor t = add_cpu(a, b);

    if (t.requires_grad()) {
      t.set_ctx({a, b}, backward_add_cpu);
    }

    return t;
}

Tensor AddImpl::gpu(const Tensor& a, const Tensor& b) {
    Tensor t = add_gpu(a, b);

    if (t.requires_grad()) {
      t.set_ctx({a, b}, backward_add_gpu);
    }

    return t;
}

