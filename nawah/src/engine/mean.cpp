#include "tensor.h"
#include "engine/ops.h"
#include "engine/ops/impl/mean.h"

Tensor MeanImpl::cpu(const Tensor& a, int dim, bool keepdim) {
    return mean_cpu(a, dim, keepdim);
}

Tensor MeanImpl::gpu(const Tensor& a, int dim, bool keepdim) {
    return mean_gpu(a, dim, keepdim);
}



