#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/matmul.h"

Tensor MatmulImpl::cpu(const Tensor& a, const Tensor& b) {
    Tensor t = matmul_cpu(a, b);


    return t;
}

Tensor MatmulImpl::gpu(const Tensor& a, const Tensor& b) {
    Tensor t = matmul_gpu(a, b);


    return t;
}

