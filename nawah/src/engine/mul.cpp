#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/mul.h"
#include "autograd/bmul.h"

Tensor MulImpl::cpu(const Tensor& a, const Tensor& b) {
    Tensor t = mul_cpu(a, b);

    if (t.requires_grad()) {
      t.set_ctx({a, b}, backward_mul_cpu);
    }

    return t;
}

Tensor MulImpl::gpu(const Tensor& a, const Tensor& b) {
    Tensor t = mul_gpu(a, b);


    if (t.requires_grad()) {
      t.set_ctx({a, b}, backward_mul_gpu);
    }
    return t;
}

