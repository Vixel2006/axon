#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/sub.h"
#include "autograd/bsub.h"

Tensor SubImpl::cpu(const Tensor& a, const Tensor& b) {
    Tensor t = sub_cpu(a, b);


    if (t.requires_grad()) {
      t.set_ctx({a, b}, backward_sub_cpu);
    }

    return t;
}

Tensor SubImpl::gpu(const Tensor& a, const Tensor& b) {
    Tensor t = sub_gpu(a, b);

    if (t.requires_grad()) {
      t.set_ctx({a, b}, backward_sub_gpu);
    }

    return t;
}

