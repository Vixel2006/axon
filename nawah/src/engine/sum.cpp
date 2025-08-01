#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/sum.h"

Tensor SumImpl::cpu(const Tensor& a, int dim, bool keepdim) {
    return sum_cpu(a, dim, keepdim);
}

Tensor SumImpl::gpu(const Tensor& a, int dim, bool keepdim) {
    return sum_gpu(a, dim, keepdim);
}

