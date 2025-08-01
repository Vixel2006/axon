#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/mul.h"

Tensor MulImpl::cpu(const Tensor& a, const Tensor& b) {
    Tensor t = mul_cpu(a, b);


    return t;
}

Tensor MulImpl::gpu(const Tensor& a, const Tensor& b) {
    Tensor t = mul_gpu(a, b);


    return t;
}

