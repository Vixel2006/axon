#include "tensor.h"
#include "autograd/ops.h"

void CudaAutograd::flatten(Tensor& out, std::vector<Tensor>& prev) {
  Tensor& a = prev[0];
  std::vector<int64_t> new_shape = a.shape();
  out = out.view(new_shape);
}

