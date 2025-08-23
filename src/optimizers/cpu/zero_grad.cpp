#include "optimizers.h"
#include "tensor.h"
#include <omp.h>

void CpuOptimizers::zero_grad(std::vector<std::shared_ptr<Tensor>> &params) {
  for (auto param : params) {
    auto *t = param.get();
    float *grad = static_cast<float *>(t->grad_ptr().get());
    size_t n = t->numel();

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      grad[i] = 0;
    }
  }
}
