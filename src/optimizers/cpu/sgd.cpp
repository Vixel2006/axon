#include "optimizers.h"
#include "tensor.h"
#include <omp.h>

void CpuOptimizers::SGD(std::vector<std::shared_ptr<Tensor>> &params,
                        float lr) {
  for (auto param : params) {
    auto *t = param.get();
    float *data = static_cast<float *>(t->data_ptr().get());
    float *grad = static_cast<float *>(t->grad_ptr().get());
    size_t n = t->numel();

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      data[i] -= lr * grad[i];
    }
  }
}
