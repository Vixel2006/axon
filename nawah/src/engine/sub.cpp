#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/sub.h"

Tensor SubImpl::cpu(const Tensor& a, const Tensor& b) {
    Tensor t = sub_cpu(a, b);


    return t;
}

Tensor SubImpl::gpu(const Tensor& a, const Tensor& b) {
    Tensor t = sub_gpu(a, b);


    return t;
}

